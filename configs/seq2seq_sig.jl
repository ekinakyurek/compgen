condconfig = Dict(
            "H"=>1024,
            "E"=>1024,
            "B"=>64,
            "attdim"=>1024,
            "optim"=>Adam(lr=0.0001),
            "gradnorm"=>0.0,
            "lang"=>"spanish",
            "N"=>100,
            "epoch"=>100,
            "pdrop"=>0.0,
            "Nlayers"=>1,
            "activation"=>ELU,
            "maxLength"=>45,
            "task"=>SIGDataSet,
            "patiance"=>0,
            "lrdecay"=>0.9,
            "beam_width" => 4,
            "copy" => true,
            "self_attention"=>true,
            "outdrop" => 0.0,
            "attdrop" => 0.0,
            "outdrop_test" => false,
            "positional" => false,
            "condmodel"=>Seq2Seq,
            "subtask"=>"analyses",
            "split"=>"jacob",
            "paug"=>.0,
            "model"=>Recombine,
            "conditional"=>true,
            "n_epoch"=>100,
            "bestval"=>false,
            "gamma"=>0.1
            )
