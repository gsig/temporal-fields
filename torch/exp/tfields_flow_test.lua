--  Action recognition experiment
-- 
--  Purpose: 
--  
--  start fbth
--  Usage: dofile '../exp/test1.lua'

local info = debug.getinfo(1,'S');
name = info.source
name = string.sub(name,1,#name-4) --remove ext
local name = name:match( "([^/]+)$" ) --remove folders
arg = arg or {}
morearg = {
'-name',name,
'-netType','crf_atfflow',
'-dataset','charadesflow_more',
'-LR','1e-5',
'-epochSize','0.1',
'-LR_decay_freq','30',
'-testSize','0.00001',
'-nEpochs','80',
'-convLR','1',
'-nThreads','5',
'-batchSize','80',
'-accumGrad','3',
'-testOnly','true',
--'-retrain','./tfields_flow.t7', -- path to the trained model to use
'-goals','30', 
'-videoSize','0.1',
'-videoSizeGoal','0.1',
'-intent_decay', '1e-3',
'-messageDecay','0.3',
'-marginal','max',
'-cacheDir','/mnt/raid00/gunnars/cache/',
'-data','/mnt/raid00/gunnars/Charades_v1_flow/',
'-actionmap','../Charades_v1_mapping.txt',
'-trainfile','../Charades_v1_train.csv',
'-testfile','../Charades_v1_test.csv',
'-pretrainpath','../',
'-optnet','true',
}
for _,v in pairs(morearg) do
    table.insert(arg,v)
end
dofile 'main.lua'
