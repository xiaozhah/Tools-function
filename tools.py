# -*- coding: utf-8 -*
import os,sys,shutil
import numpy as np
import random,struct
import multiprocessing,h5py
from tqdm import tqdm
from scipy.stats.stats import pearsonr
from glob import glob
from subprocess import Popen

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, 'wt')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)  
        self.log.flush()        
    
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.log.close()
        sys.stdout=self.terminal
        pass 

def SaveMkdir(dir):
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
    except:
        os.makedirs(dir)

def delDir(dir):
    shutil.rmtree(dir)

def durDir_2_cutedlabDir(ori_Dir,trg_Dir,ref_file=None):
    SaveMkdir(trg_Dir)
    if ref_file==None: 
        ref_list = map(lambda i:os.path.split(i)[1].split('.')[0], glob(os.path.join(ori_Dir,'*.dur')))
    else:
        ref_list = np.loadtxt(ref_file,'str')
    for file in glob(os.path.join(ori_Dir,'*.dur')):
        name = os.path.split(file)[1].split('.')[0]
        print "transfering %s" % name
        if name in ref_list:
            ori_name = os.path.join(ori_Dir,name+'.dur')
            trg_name = os.path.join(trg_Dir,name+'.lab')
            durFile_2_cutedlabFile(ori_name,trg_name)

def durFile_2_cutedlabFile(ori_file,trg_file):
    table = map(lambda i:i.split(),open(ori_file,'rt').readlines())
    with open(trg_file,'wt') as f:
        frame = 0
        for lis in table:
            context = lis[0]
            state = int(lis[1])
            duration = int(lis[2]) * 50000
            if state == 2:
                f.write('%d %d %s[%d] %s\n' % (frame,frame+duration,context,state,context))
            else:
                f.write('%d %d %s[%d]\n' % (frame,frame+duration,context,state))
            frame += duration

def SimpleCopyFiles(ori_Dir,trg_Dir,ref_file=None):
    SaveMkdir(trg_Dir)
    if ref_file==None: 
        ref_list = map(lambda i:i.split('.')[0],os.listdir(ori_Dir))
    else:
        ref_list = np.loadtxt(ref_file,'str')
    for file in os.listdir(ori_Dir):
        name = file.split('.')[0]
        if name in ref_list:
            print "copying %s" % name
            ori_name = os.path.join(ori_Dir,file)
            trg_name = os.path.join(trg_Dir,file)
            shutil.copyfile(ori_name,trg_name)

# 针对目录计算均值数据
def cal_MeanStd(datadir, dim, ref_file=None):
    # This method is efficient for large datadir
    # First row is mean vector
    # Second row is std vector
    tqdm.write('Calculate MeanStd Mean File...')
    files = os.listdir(datadir) 
    if ref_file!=None: 
        ref_list = np.loadtxt(ref_file,'str')
        files = [file for file in files if file.split('.')[0] in ref_list]
    filenum = len(files)
    mean_std = np.zeros([2,dim],dtype=np.float64)
    file_mean = np.zeros([filenum,dim+1],dtype=np.float64)
    file_std = np.zeros([filenum,dim+1],dtype=np.float64)
    for i in tqdm(range(len(files))):     
        file = datadir+os.sep+files[i]
        data = ReadFloatRawMat(file,dim)
        file_mean[i][0] = data.shape[0]
        file_std[i][0] = data.shape[0]      
        file_mean[i][1:] = np.mean(data,0)
        file_std[i][1:] = np.mean(data**2,0)
    
    file_sum = (file_mean[:,0]*file_mean[:,1:].T).T 
    file_ssum = (file_std[:,0]*file_std[:,1:].T).T 
    
    mean_std[0] = np.sum(file_sum,0) / np.sum(file_mean[:,0])   
    mean_std[1] = np.sqrt(np.sum(file_ssum,0)/ np.sum(file_mean[:,0]) - mean_std[0]**2)
    return mean_std

def cal_MinMax(datadir, dim, ref_file=None):
    #First row is min vector
    #Second row is max vector
    tqdm.write('Calculate MinMax Mean File...')
    files = os.listdir(datadir)
    if ref_file!=None: 
        ref_list = np.loadtxt(ref_file,'str')
        files = [file for file in files if file.split('.')[0] in ref_list]
    filenum = len(files)
    min_max = np.zeros([2,dim],dtype=np.float64)
    for i in tqdm(range(len(files))):
        file = datadir+os.sep+files[i]
        data = ReadFloatRawMat(file,dim)
        if i==0:
            min_max[0]=np.min(data,axis=0)
            min_max[1]=np.max(data,axis=0)
            continue
        min_max[0] = np.min([min_max[0], np.min(data,axis=0)],axis=0)
        min_max[1] = np.max([min_max[1], np.max(data,axis=0)],axis=0)
    return min_max

# 针对目录计算*.mean文件且新建一个归一化文件夹
def Normalization_MeanStd_Dir(datadir,dim,norm_dim,mean_data=None,name='',prop='',ref_file=None):
    #name is model's name
    #prop may be data or label.
    if np.all(mean_data) == None:
        mean_std = cal_MeanStd(datadir,dim,ref_file)
        WriteArrayFloat(os.path.join(datadir,os.pardir,'MeanStd_'+name+'_'+prop+'.mean'),mean_std)
    else:
        assert(mean_data.shape[1]==dim)
        mean_std = mean_data
    
    outdir = datadir+'_normalization'
    SaveMkdir(outdir)
    for line in sorted(os.listdir(datadir)):
        print 'normalizing: file %s'%line
        infile  = datadir+os.sep+line
        outfile = outdir +os.sep+line
        indata  = ReadFloatRawMat(infile,dim)
        outdata = Normalization_MeanStd(indata,norm_dim,mean_std)
        WriteArrayFloat(outfile,outdata)
    #shutil.rmtree(datadir)

def Normalization_MinMax_Dir(datadir,dim,norm_dim,mean_data=None,name='',prop='',ref_file=None):
    #name is model's name
    #prop may be data or label.
    if np.all(mean_data) == None:
        min_max=cal_MinMax(datadir,dim,ref_file)
        WriteArrayFloat(os.path.join(datadir,os.pardir,'MinMax_'+name+'_'+prop+'.mean'),min_max)
    else:
        assert(mean_data.shape[1]==dim)
        min_max = mean_data

    print '123'
    outdir = datadir+'_normalization'
    SaveMkdir(outdir)
    for line in sorted(os.listdir(datadir)):
        print 'normalizing: file %s'%line
        infile  = datadir+os.sep+line
        outfile = outdir +os.sep+line
        indata  = ReadFloatRawMat(infile,dim)
        outdata = Normalization_MinMax(indata,norm_dim,min_max)
        WriteArrayFloat(outfile,outdata)
    #shutil.rmtree(datadir)

# 根据均值数据计算矩阵的归一化结果
def Normalization_MinMax(indata,norm_dim,meanData):
    nframes = indata.shape[0]
    assert(indata.shape[1]==meanData.shape[1])
    outdata = indata
    temp1=np.tile(meanData[0][norm_dim],(nframes,1))
    temp2=np.tile(meanData[1][norm_dim],(nframes,1))-np.tile(meanData[0][norm_dim],(nframes,1))
    outdata[:,norm_dim] = (indata[:,norm_dim] - temp1) / temp2
    return outdata

def Normalization_MeanStd(indata,norm_dim,meanData):
    nframes = indata.shape[0]
    assert(indata.shape[1]==meanData.shape[1])
    outdata = indata
    mean = np.tile(meanData[0][norm_dim],(nframes,1))
    std  = np.tile(meanData[1][norm_dim],(nframes,1))
    outdata[:,norm_dim] = (indata[:,norm_dim] - mean) / std
    return outdata

# 根据均值数据计算矩阵的反归一化结果
def DeNormalization_MinMax(indata,norm_dim,meanData):
    nframes = indata.shape[0]
    outdata = indata
    temp1=np.tile(meanData[1][norm_dim],(nframes,1))-np.tile(meanData[0][norm_dim],(nframes,1))
    temp2=np.tile(meanData[0][norm_dim],(nframes,1))
    outdata[:,norm_dim] = indata[:,norm_dim] * temp1 + temp2
    return outdata

def DeNormalization_MeanStd(indata,norm_dim,meanData):
    nframes = indata.shape[0]
    outdata = indata
    mean = np.tile(meanData[0][norm_dim],(nframes,1))
    std  = np.tile(meanData[1][norm_dim],(nframes,1))
    outdata[:,norm_dim] = indata[:,norm_dim] * std + mean
    return outdata

def SaveRun(cmd):
    assert(os.system(cmd)==0)

def ParallelRun(cmds,threads=5):    
    if threads>multiprocessing.cpu_count():
        threads=multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=threads)
    pool.map(SaveRun, cmds)
    pool.close()
    pool.join()

def MultiRun(cmds):
    processes = [Popen(cmd, shell=True) for cmd in cmds]
    for process in processes:
        process.wait()

def ReadFloatRawMat(datafile,column):
    data = np.fromfile(datafile,dtype=np.float32)
    assert len(data)%column == 0, 'ReadFloatRawMat %s, column wrong!'%datafile
    assert len(data) != 0, 'empty file: %s'%datafile
    data.shape = [len(data)/column, column]
    return np.float64(data)

def WriteArrayFloat(file,data):
    tmp=np.array(data,dtype=np.float32)
    tmp.tofile(file)

def WriteArrayDouble(file,data):
    tmp=np.array(data,dtype=np.float64)
    tmp.tofile(file)

def read_file_list(file_name):
    return np.loadtxt(file_name,'str')

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)
    return  file_name_list
    
# compute distortion
def compute_mse(ref_data, gen_data):
    diff = (ref_data - gen_data) ** 2
    sum_diff = np.sum(diff, axis=1)
    sum_diff = np.sqrt(sum_diff)       
    sum_diff = np.sum(sum_diff, axis=0)
    return  sum_diff

def compact_to_single_binary_file(data_Dir, label_Dir, data_dims, label_dims, filelst, output_file):
    fout = open(output_file,'wb')   
    for name in read_file_list(filelst):
        name = name+'.dat'
        print 'Processing file %s'%name
        data1 = ReadFloatRawMat(os.path.join(data_Dir, name), data_dims)
        data2 = ReadFloatRawMat(os.path.join(label_Dir,name), label_dims)
        data=np.concatenate((data1,data2),axis=1).flatten()
        fout.write(struct.pack('<'+str(data.size)+'f',*data.tolist()))
    fout.close()

def transfer_binary_file_to_hdf5(binary_file, hdf5_file, name, data_dims, label_dims):
    float_size=4
    input_and_output_node=data_dims+label_dims

    with open(binary_file,'rb') as f:
        f.seek(0,os.SEEK_END)
        file_len=f.tell()/(float_size*input_and_output_node)
        print "row number is "+str(file_len)
        print 'the size of binary_file is '+str(os.path.getsize(binary_file)/(1024**3.))+' GB'

    with h5py.File(hdf5_file, "w") as f:
        linguistic =  f.create_dataset(name[0], (file_len, data_dims),dtype='float', chunks=True)
        output = f.create_dataset(name[1], (file_len,label_dims),dtype='float', chunks=True)
        fin=open(binary_file,'rb')
        index=range(file_len)
        random.shuffle(index)
        for i in tqdm(xrange(file_len)):
            #Shuffle the index
            linguistic [index[i]]  = np.array(struct.unpack('<'+str(data_dims)+'f',fin.read(float_size*data_dims)))
            output[index[i]]  = np.array(struct.unpack('<'+str(label_dims)+'f',fin.read(float_size*label_dims)))
        fin.close()
    os.remove(binary_file)

def compute_f0_mse(ref_data, gen_data):
    ref_vuv_vector = np.zeros((ref_data.size, 1))
    gen_vuv_vector = np.zeros((ref_data.size, 1))
    
    ref_vuv_vector[ref_data > 0.0] = 1.0
    gen_vuv_vector[gen_data > 0.0] = 1.0

    sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
    voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
    voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
    voiced_frame_number = voiced_gen_data.size

    f0_mse = (voiced_ref_data - voiced_gen_data) ** 2
    f0_mse = np.sum((f0_mse))

    vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
    vuv_error = np.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])

    return  f0_mse, vuv_error, voiced_frame_number

def compute_corr(ref_data, gen_data):
    return pearsonr(ref_data, gen_data)[0]

def compute_f0_corr(ref_data, gen_data):
    ref_vuv_vector = np.zeros((ref_data.size, 1))
    gen_vuv_vector = np.zeros((ref_data.size, 1))
    
    ref_vuv_vector[ref_data > 0.0] = 1.0
    gen_vuv_vector[gen_data > 0.0] = 1.0

    sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
    voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
    voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
    f0_corr = compute_corr(voiced_ref_data, voiced_gen_data)

    return f0_corr

def MCD(reference_list, generation_list,feature_dim):
    number=len(reference_list)
    distortion = 0.0
    total_frame_number = 0
    print('Evaluating MCD')
    for i in xrange(number):
        if(i % 1000)==0:
            print('%.2f'%(float(i)/number*100)+'%')
        ref_data = ReadFloatRawMat(reference_list[i],feature_dim).astype(np.float32)
        gen_data=ReadFloatRawMat(generation_list[i],feature_dim).astype(np.float32)
        ref_frame_number=min(ref_data.size/feature_dim,gen_data.size/feature_dim)
        temp_distortion = compute_mse(ref_data[0:ref_frame_number, 0:feature_dim], gen_data[0:ref_frame_number, 0:feature_dim])
        distortion += temp_distortion
        total_frame_number += ref_frame_number
    distortion /= float(total_frame_number)
    distortion*= (10 /np.log(10)) * np.sqrt(2.0)
    return distortion

def F0_VUV_distortion(reference_list, generation_list):
    number=len(reference_list)
    total_voiced_frame_number = 0
    vuv_error = 0
    distortion = 0.0
    total_frame_number = 0
    ref_all_files_data = np.reshape(np.array([]), (-1,1))
    gen_all_files_data = np.reshape(np.array([]), (-1,1))
    print('Evaluating F0_VUV_distortion')
    for i in xrange(number):
        if(i % 1000)==0:
            print('%.2f'%(float(i)/number*100)+'%')
        length1=len(open(reference_list[i],'rt').readlines())
        length2=len(open(generation_list[i],'rt').readlines())
        length=min(length1,length2)
        f0_ref=np.zeros(length).reshape(length,1)
        f0_gen=np.zeros(length).reshape(length,1)
        count_ref=0
        count_gen=0
        for line in open(reference_list[i],'rt'):
            if count_ref>0:
                f0_ref[count_ref-1,0]=float(line.split()[0])
                if count_ref==length:
                    break
            count_ref+=1
        for line in open(generation_list[i],'rt'):
            f0_gen[count_gen,0]=float(line.split()[0])
            if count_gen==length-1:
                break
            count_gen+=1
        ref_all_files_data = np.concatenate((ref_all_files_data, f0_ref), axis=0)
        gen_all_files_data = np.concatenate((gen_all_files_data, f0_gen), axis=0)
        temp_distortion, temp_vuv_error, voiced_frame_number = compute_f0_mse(f0_ref, f0_gen)
        vuv_error += temp_vuv_error
        total_voiced_frame_number += voiced_frame_number
        distortion += temp_distortion
        total_frame_number += length
    distortion /= float(total_voiced_frame_number)
    vuv_error  /= float(total_frame_number)
    distortion = np.sqrt(distortion)
    f0_corr = compute_f0_corr(ref_all_files_data, gen_all_files_data)
    return  distortion, f0_corr, vuv_error

if __name__=='__main__':
    pass
    #durFile_2_cutedlabFile(r"F:\xzhou\Yanping_13k\gen\yanping\sd\00000002.dur",'test.txt')
