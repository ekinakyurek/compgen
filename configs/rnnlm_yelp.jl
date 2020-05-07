config = Dict(
               "model"=> RNNLM,
               "lang"=>"turkish",
               "kill_edit"=>false,
               "attend_pr"=>0,
               "A"=>256,
               "H"=>300,
               "Z"=>64,
               "E"=>300,
               "B"=>128,
               "attdim"=>128,
               "concatz"=>true,
               "optim"=>Adam(lr=0.001),
               "kl_weight"=>0.0,
               "kl_rate"=> 0.05,
               "fb_rate"=>4,
               "N"=>10000,
               "useprior"=>true,
               "aepoch"=>1, #20
               "epoch"=>8,  #40
               "Ninter"=>10,
               "pdrop"=>0.1,
               "calctrainppl"=>false,
               "Nsamples"=>100,
               "pplnum"=>1000,
               "authresh"=>0.1,
               "Nlayers"=>3,
               "Kappa"=>25,
               "max_norm"=>10.0,
               "eps"=>1.0,
               "activation"=>ELU,
               "maxLength"=>25,
               "calc_trainppl"=>false,
               "num_examplers"=>2,
               "dist_thresh"=>0.5,
               "max_cnt_nb"=>10,
               "task"=>YelpDataSet,
               "patiance"=>4,
               "lrdecay"=>0.5,
               "conditional" => false,
               "split" => "simple",
               "splitmodifier" => "right",
               "beam_width" => 4,
               "copy" => false,
               "writedrop" => 0.1,
               "attdrop" => 0.1,
               "insert_delete_att" =>false,
               "p(xp=x)" => 0.1
               )
