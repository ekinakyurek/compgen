config = Dict(
             "model"=> Recombine,
             "lang"=>"spanish",
             "kill_edit"=>true,
             "attend_pr"=>0,
             "A"=>4,
             "H"=>768,
             "Z"=>2,
             "E"=>32,
             "B"=>16,
             "attdim"=>128,
             "Kpos" =>16,
             "concatz"=>true,
             "optim"=>Adam(lr=0.0001),
             "gradnorm"=>0.0,
             "kl_weight"=>0.0,
             "kl_rate"=> 0.05,
             "fb_rate"=>4,
             "N"=>180,
             "Nsamples"=>180,
             "aepoch"=>1, #20
             "epoch"=>25,  #40
             "Ninter"=>10,
             "pdrop"=>0.5,
             "calctrainppl"=>false,
             "pplnum"=>1000,
             "authresh"=>0.1,
             "Nlayers"=>1,
             "Kappa"=>1.0,
             "max_norm"=>1.0,
             "eps"=>0.9,
             "activation"=>ELU,
             "maxLength"=>45,
             "calc_trainppl"=>false,
             "num_examplers"=>2,
             "dist_thresh"=>0.5,
             "max_cnt_nb"=>5,
             "task"=>SIGDataSet,
             "patiance"=>6,
             "lrdecay"=>0.5,
             "conditional" => true,
             "split" => "medium",
             "splitmodifier" => "jump",
             "beam_width" => 4,
             "copy" => true,
             "writedrop" => 0.1,
             "outdrop" => 0.5,
             "attdrop" => 0.1,
             "outdrop_test" => false,
             "positional" => true,
             "masktags" => false,
             "condmodel"=>Seq2Seq,
             "rwritedrop"=>0.0,
             "rpatiance"=>0,
             "subtask"=>"reinflection",
             "paug"=>0,
             "seperate"=>true,
             "feedcontext"=>true,
             "path"=>"jacob/morph/",
             "seperate_emb"=>true,
             "hints"=>4,
             "seed"=>0,
             "self_attention"=>true,
             "nproto"=>2,
             "temp"=>0.05
             )
