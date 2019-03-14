# -*- coding: utf-8 -*
from tools import *
import re
from multiprocessing import Pool
import platform

toolsDir = os.path.dirname(os.path.abspath(__file__))

mlpg_tool = os.path.join(toolsDir, 'bin','mlpg_mv.exe')
straight  = os.path.join(toolsDir, 'bin','straight_mceplsf.exe')

winfile = os.path.join(toolsDir, '../win/mcep_d1.win')
accwinfile =os.path.join(toolsDir, '../win/mcep_d2.win')

if platform.system() == 'Linux':
    mlpg_tool = 'wine ' + mlpg_tool
    straight  = 'wine ' + straight

perlfile = os.path.join(toolsDir, 'predict_pitch.pl')       

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

def DNN_MLPG(dir,dim,meanFile,list_file_name=None,cal_GV=False,syn=False,
             compute_distortion=False,ref_cep_dir=None,ref_f0_dir=None,
             mode = '', MLPG = True):
    mean_std = ReadFloatRawMat(meanFile,dim)
    variance = mean_std[1]**2
    speDim = (dim-4)/3

    f0Dir = dir+os.sep+'LF0'
    SpeDir = dir+os.sep+'SPE'
    UVDir = dir+os.sep+'UV'
    tmpDir = dir+os.sep+'tmp'
    outDir_f0 = dir+os.sep+'gen_f0'
    outDir_cep = dir+os.sep+'gen_cep'
    SaveMkdir(tmpDir)    
    SaveMkdir(outDir_f0)
    SaveMkdir(outDir_cep)
    
    if MLPG == True:
        cmd1s = []
        cmd2s = []
        for name in list_file_name:
            print 'DNN_MLPG ',name
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
            outCepFile = outDir_cep+os.sep+name+'.cep'

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
    
    if cal_GV:
        outDir_cep_addGV = dir+os.sep+'gen_cep+GV'
        SaveMkdir(outDir_cep_addGV)

        for name in list_file_name:
            print name
            outCepFile = outDir_cep+os.sep+name+'.cep'
            data = ReadFloatRawMat(outCepFile, speDim)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            meanData = np.vstack((mean,std))
            data = Normalization_MeanStd(data,range(speDim),meanData)
            data = DeNormalization_MeanStd(data,range(speDim),mean_std)

            outCepFile_addGV = outDir_cep_addGV+os.sep+name+'.cep'
            WriteArrayFloat(outCepFile_addGV,data)
    
    if compute_distortion==True:
        if cal_GV:
            generation_cep_list=prepare_file_path_list(read_file_list(list_file_name), outDir_cep_addGV, '.cep')
        else:
            generation_cep_list=prepare_file_path_list(read_file_list(list_file_name), outDir_cep, '.cep')
        reference_cep_list=prepare_file_path_list(read_file_list(list_file_name), ref_cep_dir, '.cep')
        reference_f0_list=prepare_file_path_list(read_file_list(list_file_name), ref_f0_dir, '.f0')
        generation_f0_list=prepare_file_path_list(read_file_list(list_file_name), outDir_f0, '.f0')
        MCD_value=MCD(reference_cep_list, generation_cep_list,speDim)
        f0_rmse, f0_corr, vuv_error=F0_VUV_distortion(reference_f0_list, generation_f0_list)
        print dir
        print 'MCD: %.4f dB; F0:- RMSE: %.4f Hz; CORR: %.4f; VUV: %.4f%%' \
                        %(MCD_value, f0_rmse, f0_corr, vuv_error*100.)

    if syn==True:
        print dir
        if cal_GV:
            outDir_wav = dir+os.sep+'gen_wav+GV'
        else:
            outDir_wav = dir+os.sep+'gen_wav'
        SaveMkdir(outDir_wav)
        cmds = []
        if mode == 'Calculate reconstruction error':
            list_file_name = list_file_name[:10]
        for name in list_file_name:
            if cal_GV:
                outCepFile = outDir_cep_addGV+os.sep+name+'.cep'
            else:
                outCepFile = outDir_cep+os.sep+name+'.cep'
            outF0File = outDir_f0+os.sep+name+'.f0'
            outWavFile = outDir_wav+os.sep+name+'.wav'    
            cmd = r'%s -mcep -pow -order %d -shift 5 -f0file %s -syn %s %s'%(straight,speDim-1,outF0File,outCepFile,outWavFile)    
            cmds.append(cmd)
        ParallelRun(cmds, threads=32)

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

# 将计算得到的反归一化6-dims时长（小数）转化为整数
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
