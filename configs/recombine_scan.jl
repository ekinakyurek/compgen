config = Dict(
              "model"=> Recombine,
              "lang"=>"turkish",
              "kill_edit"=>false,
              "attend_pr"=>0,
              "A"=>32,
              "H"=>512,
              "Z"=>16,
              "E"=>64,
              "B"=>64,
              "attdim"=>128,
              "Kpos" =>16,
              "concatz"=>true,
              "optim"=>Adam(lr=0.001),
              "gradnorm"=>1.0,
              "kl_weight"=>0.0,
              "kl_rate"=> 0.05,
              "fb_rate"=>4,
              "N"=>400,
              "useprior"=>true,
              "aepoch"=>1, #20
              "epoch"=>8 ,  #40
              "Ninter"=>10,
              "pdrop"=>0.5,
              "calctrainppl"=>false,
              "Nsamples"=>500,
              "pplnum"=>1000,
              "authresh"=>0.1,
              "Nlayers"=>1,
              "Kappa"=>25,
              "max_norm"=>10.0,
              "eps"=>1.0,
              "activation"=>ELU,
              "maxLength"=>45,
              "calc_trainppl"=>false,
              "num_examplers"=>2,
              "dist_thresh"=>0.5,
              "max_cnt_nb"=>5,
              "task"=>SCANDataSet,
              "patiance"=>4,
              "lrdecay"=>0.5,
              "conditional" => true,
              "split" => "add_prim",
              "splitmodifier" => "jump",
              "beam_width" => 4,
              "copy" => true,
              "writedrop" => 0.5,
              "outdrop" => 0.7,
              "attdrop" => 0.0,
              "outdrop_test" => true,
              "positional" => true,
              "masktags" => false,
              "condmodel"=>Seq2Seq,
              "subtask"=>nothing,
              "paug"=>0.01,
              "seperate"=>true,
              "seperate_emb"=>false,
              "feedcontext"=>true,
              "seed"=>1,
              "self_attention"=>false,
              "use_insert_delete"=>false,
              "p(xp=x)" => 0.01,
              "nproto" => 2,
              "temp" => 0.2,
              "beam" => true,
              "rare_token"=>false,
              "modeldir"=>nothing
              )
