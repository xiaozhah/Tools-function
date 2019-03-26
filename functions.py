# -*- coding: utf-8 -*
from tools import *
import re
from multiprocessing import Pool
import platform
from fastdtw import fastdtw,dtw
from scipy.spatial.distance import euclidean
import uuid

toolsDir = os.path.dirname(os.path.abspath(__file__))

mlpg_tool     = os.path.join(toolsDir, 'bin','mlpg_mv.exe')
straight      = os.path.join(toolsDir, 'bin','straight_mceplsf.exe')
F0PostProcess = os.path.join(toolsDir, 'bin','F0PostProcess.exe')
Freq2lFreq    = os.path.join(toolsDir, 'bin','Freq2lFreq.exe')

winfile       = os.path.join(toolsDir, '..', 'win', 'mcep_d1.win')
accwinfile    = os.path.join(toolsDir, '..', 'win', 'mcep_d2.win')

if platform.system() == 'Linux' or platform.system() == 'Darwin':
    mlpg_tool = 'wine ' + mlpg_tool
    straight  = 'wine ' + straight
    F0PostProcess = 'wine ' + F0PostProcess
    Freq2lFreq = 'wine ' + Freq2lFreq

perlfile = os.path.join(toolsDir, 'predict_pitch.pl')

def RmSilenceParallel(AlignmentDir, SpeDir, spedim, ratio, phoneme = 'sil', threads = 5):
    SpeDir_dropSil = SpeDir+'_dropSil'
    SaveMkdir(SpeDir_dropSil)   

    print 'Removing silence: %s'%SpeDir
    argList = []
    for file in os.listdir(SpeDir):
        name = file.split('.')[0]   
        spefile = SpeDir+os.sep+name+'.dat'
        labelfile = AlignmentDir+os.sep+name+'.lab'
        spe_dropSil_file = SpeDir_dropSil+os.sep+name+'.dat'        
        argList.append([labelfile,spefile, spedim, spe_dropSil_file,ratio,phoneme])       
    pool = Pool(threads)
    pool.map(RmSilenceSingleFile,argList)
    pool.close()
    pool.join()

def RmSilenceSingleFile(args):
    [labelfile,spefile, spedim, spe_dropSil_file,ratio,phoneme] = args
    
    silPos = []
    lines = open(labelfile,'rt').readlines() 
    for i in range(len(lines)):
        if lines[i]=='':continue
        if lines[i].split()[2].endswith('[2]'):
            string=lines[i].split()[2]
            phone_name=string[string.find('-')+1:string.find('+')]
            if(phone_name==phoneme):
                phone_start = int(round(float(lines[i].split()[0])/50000.0))
                phone_end = int(round(float(lines[i+4].split()[1])/50000.0))
                silPos.append([phone_start,phone_end])

    temp1 = range(silPos[0][0],int(silPos[0][0]+ratio*(silPos[0][1]-silPos[0][0])))
    temp2 = range(int(silPos[1][0]+(1-ratio)*(silPos[1][1]-silPos[1][0])),silPos[1][1])
    rmSilPos = temp1+temp2       

    ######## removing silence from two files    
    speData = ReadFloatRawMat(spefile,spedim)
    if rmSilPos==[]:
        newSpeData = speData
    else:
        newSpeData = np.delete(speData,rmSilPos,0)
    WriteArrayFloat(spe_dropSil_file,newSpeData)        

#计算矩阵(M * N)的一二阶差分，构建更多特征的矩阵(M * 3N)
def calDynamicData(data):
    dData = np.zeros(data.shape)
    ddData = np.zeros(data.shape)
    
    win1 = [-0.5,0,0.5]
    win2 = [0.25,-0.5,0.25]        
    for k in range(data.shape[1]):
        dData[:,k] = np.convolve(data[:,k],np.flipud(win1),mode='same')
        ddData[:,k] = np.convolve(data[:,k],np.flipud(win2),mode='same')
    dData[0] = dData[1]
    ddData[0] = ddData[1]
    dData[-1] = dData[-2]
    ddData[-1] = ddData[-2]    

    return np.concatenate((data,dData,ddData),axis=1)    

#将频谱基频文件融合形成一个更长的特征
def F0Spe2DNNOutput(f0Dir,speDir,speDim,outDir,ref_file=None,cal_dynamic_feas=True):
    tmpDir = 'tmp'
    SaveMkdir(tmpDir)
    SaveMkdir(outDir)
    if ref_file==None: 
        ref_list = map(lambda i:i.split('.')[0],os.listdir(f0Dir))
    else:
        ref_list = np.loadtxt(ref_file,'str')
    for name in sorted(ref_list):
        print "process %s" % name            
        ####### F0 interpolation ######
        rawF0File = f0Dir+os.sep+name+'.f0'
        tmpF0File = tmpDir+os.sep+name+'.if0'  #包含了所有频率信息，只有rawF0File的第一列 和rawF0File行数相同
        oF0File = tmpDir+os.sep+name+'.of0'    #包含了插值后的频率信息 和rawF0File行数相同 插值后文件中没有零频率
        ftmp = open(tmpF0File,'wt')
        uv = []
        for line in open(rawF0File,'rt'):
            if not line.endswith('100.0\n'):continue
            if line.startswith('0.0'):
                uv += [0]  #清音则回答0
            else: 
                uv += [1]  #浊音则回答1
            ftmp.write(line.split()[0]+'\n')
        ftmp.close()     #此时rawF0File中有所有频率信息，包括0频率
        
        os.system('perl %s %s %s'%(perlfile,tmpF0File,oF0File))
        ###################################
        
        ####### interpolated F0 and spectral feature, cal dynamics, to DNN output ###################
        speFile = speDir+os.sep+name+'.cep'
        DNNOutFile = outDir+os.sep+name+'.dat'
        
        spe = ReadFloatRawMat(speFile,speDim)
        if cal_dynamic_feas:
            dspe = calDynamicData(spe)
        else:
            dspe = spe
        
        #对插值出的频率取对数运算得到f0
        f0 = np.log(np.array(map(lambda x: float(x.strip()),open(oF0File,'rt').read().splitlines())))
        f0.shape = f0.shape[0],1   #将f0变成列向量
        
        if cal_dynamic_feas:
            df0 = calDynamicData(f0)
        else:
            df0 = f0
        UV = np.array(uv,dtype=np.float64).reshape(len(uv),1)#将uv变成列向量

        #行数对齐，参考较小的行数
        if spe.shape[0]>f0.shape[0]:
            dspe = dspe[:f0.shape[0]]
        elif spe.shape[0]<f0.shape[0]:
            df0 = df0[:spe.shape[0]]
            UV = UV[:spe.shape[0]]    
        
        #将频谱的零一二阶查分，f0插值后的log的零一二阶查分，f0对清浊音的回答合并
        DNNOut = np.concatenate((dspe,df0,UV),axis=1)
        WriteArrayFloat(DNNOutFile,DNNOut)    
    shutil.rmtree(tmpDir)

#输入是f0Dir和SpeDir文件夹，输出是outDir_f0和outDir_cep文件夹
#MPLG根据最大似然准则对输入做了去动态参数的操作，静态特征更可靠
#注意：HMM的HMGenS生成的是mcep文件，因此gen_cep文件夹的文件后缀也改为mcep，以实现兼容
def MLPG(mydir, dim, meanFile, list_file_name):
    mean_std = ReadFloatRawMat(meanFile,dim)
    variance = mean_std[1]**2
    speDim = (dim-4)/3

    f0Dir = mydir+os.sep+'LF0'
    SpeDir = mydir+os.sep+'SPE'
    UVDir = mydir+os.sep+'UV'
    assert(os.path.exists(f0Dir))
    assert(os.path.exists(SpeDir))
    assert(os.path.exists(UVDir))

    tmpDir = mydir+os.sep+'tmp'
    
    outDir_f0 = mydir+os.sep+'gen_f0'
    outDir_cep = mydir+os.sep+'gen_cep'
    SaveMkdir(tmpDir)    
    SaveMkdir(outDir_f0)
    SaveMkdir(outDir_cep)
    cmd1s = []
    cmd2s = []
    for name in tqdm(list_file_name):
        tqdm.write('MLPG %s' % name)
        dspe = ReadFloatRawMat(SpeDir+os.sep+name+'.spe',speDim*3)
        dLF0 = ReadFloatRawMat(f0Dir+os.sep+name+'.lf0',3)
        uv = ReadFloatRawMat(UVDir+os.sep+name+'.uv',1).flatten()
        
        dspe_var = np.tile(variance[:speDim*3],[dspe.shape[0],1])
        lf0_var = np.tile(variance[speDim*3:-1],[dspe.shape[0],1])
        
        dspeMeanFile = tmpDir+os.sep+name+'_mean.spe'
        dLF0MeanFile = tmpDir+os.sep+name+'_mean.lf0'
        
        dspeVarFile = tmpDir+os.sep+name+'_var.spe'
        dLF0VarFile = tmpDir+os.sep+name+'_var.lf0'
        
        outF0File = outDir_f0+os.sep+name+'.f0'  
        outCepFile = outDir_cep+os.sep+name+'.mcep'

        WriteArrayDouble(dspeMeanFile,dspe)
        WriteArrayDouble(dLF0MeanFile,dLF0)
        WriteArrayDouble(dspeVarFile,dspe_var)
        WriteArrayDouble(dLF0VarFile,lf0_var)
            
        cmd1 = '%s -din -order %d -dynwinf %s -accwinf %s %s %s %s'%(mlpg_tool,1,winfile,accwinfile,dLF0MeanFile,dLF0VarFile,outF0File)
        cmd1s.append(cmd1)
        
        cmd2 = '%s -din -order %d -dynwinf %s -accwinf %s %s %s %s'%(mlpg_tool,speDim,winfile,accwinfile,dspeMeanFile,dspeVarFile,outCepFile)
        cmd2s.append(cmd2)
    
    ParallelRun(cmd1s, threads=32)
    ParallelRun(cmd2s, threads=32)
    
    for name in list_file_name:
        outF0File = outDir_f0+os.sep+name+'.f0'
        uv = ReadFloatRawMat(UVDir+os.sep+name+'.uv',1).flatten()
        f0 = np.exp(ReadFloatRawMat(outF0File,1).flatten())
        with open(outF0File,'wt') as f:
            for i in range(len(f0)):
                if uv[i]==1:
                    f.write('%f\n'%(f0[i]))
                else:
                    f.write('0\n')
    shutil.rmtree(tmpDir)

def Synthesis(mydir, dim, meanFile, list_file_name, cal_GV, mode = 'all'):
    tqdm.write('Synthesis...')
    mean_std = ReadFloatRawMat(meanFile,dim)
    
    outDir_f0 = mydir+os.sep+'gen_f0'
    outDir_cep = mydir+os.sep+'gen_cep'
    if cal_GV:
        outDir_wav = mydir+os.sep+'gen_wav+GV'
    else:
        outDir_wav = mydir+os.sep+'gen_wav'
    SaveMkdir(outDir_wav)

    if mode == 'listen few audio':
        # 只合成前十句
        list_file_name = list_file_name[:10]
    
    MLPG(mydir, dim, meanFile, list_file_name)
    assert(os.path.exists(outDir_f0))
    assert(os.path.exists(outDir_cep))

    speDim = (dim-4)/3
    
    if cal_GV:
        outDir_cep_addGV = mydir+os.sep+'gen_cep+GV'
        SaveMkdir(outDir_cep_addGV)

        for name in list_file_name:
            outCepFile = outDir_cep+os.sep+name+'.mcep'
            data = ReadFloatRawMat(outCepFile, speDim)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            meanData = np.vstack((mean,std))
            data = Normalization_MeanStd(data,range(speDim),meanData)
            data = DeNormalization_MeanStd(data,range(speDim),mean_std)

            outCepFile_addGV = outDir_cep_addGV+os.sep+name+'.mcep'
            WriteArrayFloat(outCepFile_addGV,data)
    
    cmds = []
    for name in list_file_name:
        if cal_GV:
            outCepFile = outDir_cep_addGV+os.sep+name+'.mcep'
        else:
            outCepFile = outDir_cep+os.sep+name+'.mcep'
        outF0File = outDir_f0+os.sep+name+'.f0'
        outWavFile = outDir_wav+os.sep+name+'.wav'    
        cmd = r'%s -mcep -pow -order %d -shift 5 -f0file %s -syn %s %s'%(straight,speDim-1,outF0File,outCepFile,outWavFile)    
        cmds.append(cmd)
    ParallelRun(cmds, threads=32)

def Compute_Distortion(mydir, dim, meanFile, list_file_name, ref_cep_dir, ref_f0_dir, caled_GV=False):
    outDir_f0 = mydir+os.sep+'gen_f0'
    outDir_cep = mydir+os.sep+'gen_cep'
    if caled_GV:
        outDir_cep_addGV = mydir+os.sep+'gen_cep+GV'
    
    speDim = (dim-4)/3
    MLPG(mydir, dim, meanFile, list_file_name)

    # 参考声学
    reference_cep_list=prepare_file_path_list(read_file_list(list_file_name), ref_cep_dir, '.cep')
    reference_f0_list=prepare_file_path_list(read_file_list(list_file_name), ref_f0_dir, '.f0')

    # 预测声学
    if caled_GV:
        generation_cep_list=prepare_file_path_list(read_file_list(list_file_name), outDir_cep_addGV, '.mcep')
    else:
        generation_cep_list=prepare_file_path_list(read_file_list(list_file_name), outDir_cep, '.mcep')
    generation_f0_list=prepare_file_path_list(read_file_list(list_file_name), outDir_f0, '.f0')
    
    MCD_value = MCD(reference_cep_list, generation_cep_list,speDim)
    f0_rmse, f0_corr, vuv_error=F0_VUV_distortion(reference_f0_list, generation_f0_list)
    print mydir
    print 'MCD: %.4f dB; F0:- RMSE: %.4f Hz; CORR: %.4f; VUV: %.4f%%' %(MCD_value, f0_rmse, f0_corr, vuv_error*100.)

# 根据音素时长动态调整状态时长
def getIntDurInfo(lis):
    lis = lis.astype(np.float64)
    sum_lis=np.sum(lis)
    assert(sum_lis-np.round(sum_lis)<1e-4)
    lis_acc=np.cumsum(lis)
    result=np.zeros(5)
    result[0]=np.round(lis[0])
    if(result[0]<=0):
        result[0]=1 #保证状态时长不为零
    for i in range(1,5):
        result[i]=np.round(lis_acc[i]-np.round(np.cumsum(result[:i])[-1]))
        if(result[i]<=0):
            result[i]=1 #保证状态时长不为零
    return result

# 将计算得到的模型预测的反归一化6-dims时长（小数）转化为正常6-dims整型时长
# 6-dims时长：前五维是状态时长，最后一维是音素时长
def Get_durion_readable(dur):
    dur = dur.astype(np.float64)
    state_dur_lis = np.clip(dur[:5], 1, None) # 每个状态时长至少1帧
    phone_dur = np.round(dur[-1])
    if phone_dur <= 5:
        phone_dur = np.round(np.sum(state_dur_lis)) # 每个音素时长至少5帧

    state_dur_lis = state_dur_lis/np.sum(state_dur_lis)*phone_dur    
    state_dur_lis = getIntDurInfo(state_dur_lis)
    dur_lis = np.append(state_dur_lis, np.sum(state_dur_lis)).astype(np.int)
    return dur_lis

# phone_unit_dir保存了音库单元信息，us_dir是单元挑选目录
# 需要用到其stt文件（最优备选单元信息）和trg文件(目标单元信息)
def Compute_US_ContextClasses_Diff(phone_unit_dir, us_dir, ref_file=None):
    if ref_file==None: 
        ref_list = map(lambda i:os.path.split(i)[1].split('.')[0], glob(os.path.join(us_dir,'*.stt')))
    else:
        ref_list = np.loadtxt(ref_file,'str')
    mydict = {}
    # 从phone_unit_dir文件夹读取音素单元信息便于由stt文件推断单元所属的决策树叶子节点信息
    for file in sorted(glob(os.path.join(phone_unit_dir,'*.unit'))):
        print file
        monophone = os.path.split(file)[1].split('.')[0]
        mydict[monophone] = {}
        for line in open(file,'rt'):
            if line.startswith('UnitNo'):
                UnitNo = int(line.split()[1])
            if line.startswith('CepCls'):
                CepCls = line.split()[1:]
            if line.startswith('F0Cls'):
                F0Cls  = line.split()[1:]
                mydict[monophone][UnitNo] = map(int,flatten(zip(CepCls,F0Cls)))
    
    lis = []
    files = sorted(glob(os.path.join(us_dir,'*.stt')))
    for file in files:
        name = os.path.split(file)[1].split('.')[0]
        if name in ref_list:
            print "processing %s" % name
            
            #读取备选单元
            stt_file = os.path.join(us_dir, name+'.stt')
            stt_text = open(file,'rt').readlines()
            stt_text = map(lambda i:i.split()[:2],stt_text)
            stt_text = map(lambda i:[i[0],int(i[1])],stt_text[1:])
            stt_info = []
            for i in stt_text:
                stt_info.append(mydict[i[0]][i[1]])
            
            #读取目标单元
            trg_file = os.path.join(us_dir, name+'.trg')
            trg_text = open(trg_file,'rt').readlines()
            trg_text = map(lambda i:i.split()[1:-3], trg_text[1:])
            trg_info = []
            for i in trg_text:
                trg_info.append(map(lambda i:int(i.split(',')[0]), i))

            assert(len(stt_info)==len(trg_info)) 
            for i in range(len(stt_info)):
                lis.append((np.array(stt_info[i])==np.array(trg_info[i])).astype(int).tolist())
    print ' '.join(map(lambda i:'%.2f%%' % (i*100), np.mean(lis, axis=0)))

# 需要用到其stt文件（最优备选单元信息，包含时长）和 测试集参考时长cutedlab格式
def Compute_US_Duration_RMSE(us_dir, testcutedlab_ref_dir, ref_file=None):
    if ref_file==None: 
        ref_list = map(lambda i:os.path.split(i)[1].split('.')[0], glob(os.path.join(us_dir,'*.stt')))
    else:
        ref_list = np.loadtxt(ref_file,'str')

    files = sorted(glob(os.path.join(us_dir,'*.stt')))
    stt_dur = []
    ref_dur = []
    index = 0
    index_lis = []
    pattern = re.compile('.*-(.*)\+.*')
    for file in files:
        name = os.path.split(file)[1].split('.')[0]
        if name in ref_list:
            print "processing %s" % name

            #读取备选单元
            stt_file = os.path.join(us_dir, name+'.stt')
            stt_text = open(stt_file,'rt').readlines()
            stt_text = np.array(map(lambda i:i.split()[3:10],stt_text)[1:],dtype=np.int)
            stt_dur  += map(lambda i:np.append(i[2:], i[1]-i[0]).tolist(),stt_text)
            
            #读取真实时长
            lab_file = os.path.join(testcutedlab_ref_dir, name+'.lab')
            lines = open(lab_file,'rt').read().splitlines() 
            for i in range(len(lines)):
                if lines[i]=='':continue
                if lines[i].split()[2].endswith('[2]'):
                    p = re.sub(pattern,'\\1',lines[i])
                    if p != 'sil' and p != 'sp':
                        index_lis.append(index) 
                    phone_start = int(round(float(lines[i].split()[0])/50000.0))
                    phone_end = int(round(float(lines[i+4].split()[1])/50000.0))
                    state_dur_lis = [int(round(float(lines[j].split()[1])/50000.0))-int(round(float(lines[j].split()[0])/50000.0)) for j in range(i,i+5)]
                    ref_dur += [np.append(state_dur_lis,phone_end-phone_start).tolist()]
                    index += 1
    ref_dur = np.array(ref_dur)
    stt_dur = np.array(stt_dur)
    index_lis = np.array(index_lis)
    print "All phoneme"
    print "Duration RMSE:\nS1:%f frame\nS2:%f frame\nS3:%f frame\nS4:%f frame\nS5:%f frame\nPhone:%f frame\n"\
            % tuple(np.sqrt(np.mean((ref_dur-stt_dur)**2,axis=0)))
    print "All phoneme but excluding sil & sp"
    print "Duration RMSE:\nS1:%f frame\nS2:%f frame\nS3:%f frame\nS4:%f frame\nS5:%f frame\nPhone:%f frame\n"\
            % tuple(np.sqrt(np.mean((ref_dur[index_lis]-stt_dur[index_lis])**2,axis=0)))

# 对于备选单元需要stt文件和音库的corpus_dfmcep,corpus_f0
# 对于测试集需要wav
# dfmcep文件夹（包含*.cep文件）和F0文件夹
def Compute_US_Acoustic_Distortion(us_dir, testwav_ref, dim, corpus_dfmcep, corpus_f0, ref_file=None, dtw_mode = 'fastdtw'):
    if ref_file==None: 
        ref_list = map(FileBaseName, sorted(glob(os.path.join(us_dir,'*.stt'))))
    else:
        ref_list = np.loadtxt(ref_file,'str')
    tmpDir = 'tmp_'+uuid.uuid4().hex
    SaveMkdir(tmpDir)
    ref_cep_dir=tmpDir+os.sep+'speFea_ref'
    ref_old_f0_dir =tmpDir+os.sep+'old_f0_ref'
    ref_f0_dir =tmpDir+os.sep+'f0_ref'

    align_ref_cep_dir=tmpDir+os.sep+'align_speFea_ref'
    align_ref_f0_dir =tmpDir+os.sep+'align_f0_ref'
    align_candidates_cep_dir=tmpDir+os.sep+'gen_cep'
    align_candidates_f0_dir =tmpDir+os.sep+'gen_f0'
    SaveMkdir(ref_cep_dir)
    SaveMkdir(ref_old_f0_dir)
    SaveMkdir(ref_f0_dir)
    SaveMkdir(align_ref_cep_dir)
    SaveMkdir(align_ref_f0_dir)
    SaveMkdir(align_candidates_cep_dir)
    SaveMkdir(align_candidates_f0_dir)

    corpus_files = map(FileBaseName, sorted(glob(os.path.join(corpus_dfmcep,'*.cep'))))
    test_files   = map(FileBaseName, sorted(glob(os.path.join(us_dir,'*.stt'))))
    cmds = []
    for name in test_files:
        if name in ref_list:          
            # 从testwav_ref分析其声学数据
            wavfile = os.path.join(testwav_ref, name+'.wav')
            old_f0file = os.path.join(ref_old_f0_dir, name+'.f0')
            f0file = os.path.join(ref_f0_dir, name+'.f0')
            cepfile= os.path.join(ref_cep_dir, name+'.cep')
            commandline = "%s -f 16000 -shift 5 -lf0 50 -uf0 600 -src %s %s" % (straight,wavfile,old_f0file)
            cmds.append(commandline)
    ParallelRun(cmds, threads=32)

    commandline = "%s -i %s -o %s -w %s -l %s" % (F0PostProcess,ref_old_f0_dir,ref_f0_dir,testwav_ref,ref_file)
    os.system(commandline);
    
    cmds = []
    for name in test_files:
        if name in ref_list:
            wavfile = os.path.join(testwav_ref, name+'.wav')
            f0file = os.path.join(ref_f0_dir, name+'.f0')
            cepfile= os.path.join(ref_cep_dir, name+'.cep')
            commandline = "%s -shift 5 -mcep -pow -order %d -f0file %s -ana %s %s" % (straight,dim-1,f0file,wavfile,cepfile);
            cmds.append(commandline)
    ParallelRun(cmds, threads=32)

    
    for name in tqdm(test_files):
        if name in ref_list:
            #读取备选单元
            stt_file = os.path.join(us_dir, name+'.stt')
            stt_text = open(stt_file,'rt').readlines()
            stt_synNo = map(lambda i:int(i.split()[2]), stt_text[1:])
            stt_dur = np.array(map(lambda i:i.split()[3:5],stt_text)[1:],dtype=np.int)

            # 从备选单元得到所挑选单元的声学数据
            cep_data_stt = []
            f0_data_stt = []
            for index, SynNo in enumerate(stt_synNo):
                frames_s_e = stt_dur[index]
                cep_data_stt += ReadFloatRawMat(os.path.join(corpus_dfmcep, corpus_files[SynNo]+'.cep'), dim)[frames_s_e[0]:frames_s_e[1]].tolist()
                f0_stt = open(os.path.join(corpus_f0, corpus_files[SynNo]+'.f0'), 'rt').readlines()
                f0_data_stt += map(lambda i:float(i.split()[0]), f0_stt[1:][frames_s_e[0]:frames_s_e[1]])

            cep_data_stt = np.array(cep_data_stt)
            f0_data_stt = np.array(f0_data_stt)

            f0file = os.path.join(ref_f0_dir, name+'.f0')
            cepfile= os.path.join(ref_cep_dir, name+'.cep')
            cep_data_ref = ReadFloatRawMat(cepfile, dim)
            f0_data_ref =  open(f0file, 'rt').readlines()
            f0_data_ref = np.array(map(lambda i:float(i.split()[0]),f0_data_ref[1:]))
            min_dim = min(f0_data_ref.shape[0],cep_data_ref.shape[0])
            cep_data_ref = cep_data_ref[:min_dim]
            f0_data_ref = f0_data_ref[:min_dim]

            # 归一化
            cep_data_stt_meanData = np.concatenate((np.mean(cep_data_stt,axis=0).reshape(1,-1), np.std(cep_data_stt,axis=0).reshape(1,-1)), axis=0)
            cep_data_stt = Normalization_MeanStd(cep_data_stt, range(dim), cep_data_stt_meanData)
            cep_data_ref_meanData = np.concatenate((np.mean(cep_data_ref,axis=0).reshape(1,-1), np.std(cep_data_ref,axis=0).reshape(1,-1)), axis=0)
            cep_data_ref = Normalization_MeanStd(cep_data_ref, range(dim), cep_data_ref_meanData)
            
            if dtw_mode == 'fastdtw':
                distance, path = fastdtw(cep_data_stt, cep_data_ref, dist=euclidean)
            elif dtw_mode == 'dtw':
                distance, path = dtw(cep_data_stt, cep_data_ref, dist=euclidean)
            else:
                print 'dtw_mode must be dtw or fastdtw'
                exit()
            
            cep_data_stt = cep_data_stt[map(lambda i:i[0],path)]
            f0_data_stt  = f0_data_stt[map(lambda i:i[0],path)]
            cep_data_ref = cep_data_ref[map(lambda i:i[1],path)]
            f0_data_ref  = f0_data_ref[map(lambda i:i[1],path)]
            #print data_stt.shape #(Q, 14)
            #print data_ref.shape #(Q, 14)
            assert(len(cep_data_stt)==len(cep_data_ref))
            # 反归一化
            cep_data_stt = DeNormalization_MeanStd(cep_data_stt, range(dim), cep_data_stt_meanData)
            cep_data_ref = DeNormalization_MeanStd(cep_data_ref, range(dim), cep_data_ref_meanData)

            with open(os.path.join(align_candidates_f0_dir, name + '.f0'),'wt') as f:
                for i in range(len(f0_data_stt)):
                    f.write('%f\n'%(f0_data_stt[i]))
            with open(os.path.join(align_ref_f0_dir, name + '.f0'),'wt') as f:
                for i in range(len(f0_data_ref)):
                    f.write('%f\n'%(f0_data_ref[i]))

            WriteArrayFloat(os.path.join(align_candidates_cep_dir, name + '.mcep'), cep_data_stt)
            WriteArrayFloat(os.path.join(align_ref_cep_dir, name + '.cep'), cep_data_ref)
    
    # 参考声学
    reference_cep_list=prepare_file_path_list(ref_list, align_ref_cep_dir, '.cep')
    reference_f0_list=prepare_file_path_list(ref_list, align_ref_f0_dir, '.f0')

    # 预测声学
    generation_cep_list=prepare_file_path_list(ref_list, align_candidates_cep_dir, '.mcep')
    generation_f0_list=prepare_file_path_list(ref_list, align_candidates_f0_dir, '.f0')
    
    MCD_value = MCD(reference_cep_list, generation_cep_list,dim)
    f0_rmse, f0_corr, vuv_error=F0_VUV_distortion(reference_f0_list, generation_f0_list)
    tqdm.write(us_dir)
    tqdm.write('MCD: %.4f dB; F0:- RMSE: %.4f Hz; CORR: %.4f; VUV: %.4f%%' %(MCD_value, f0_rmse, f0_corr, vuv_error*100.))
    delDir(tmpDir)
