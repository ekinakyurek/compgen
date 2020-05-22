condconfig = Dict(
              "H"=>512,
              "E"=>64,
              "B"=>64,
              "attdim"=>512,
              "optim"=>Adam(lr=0.001),
              "gradnorm"=>1.0,
              "N"=>100,
              "epoch"=>150,
              "pdrop"=>0.5,
              "Nlayers"=>1,
              "activation"=>ELU,
              "maxLength"=>45,
              "task"=>SCANDataSet,
              "patiance"=>10,
              "lrdecay"=>0.5,
              "split" => "add_prim",
              "splitmodifier" => "jump",
              "beam_width" => 4,
              "copy" => false,
              "outdrop" => 0.0,
              "attdrop" => 0.0,
              "outdrop_test" => false,
              "positional" => false,
              "condmodel"=>Seq2Seq,
              "subtask"=>nothing,
              "paug"=>0.01,
              "model"=>Recombine,
              "conditional"=>true,
              "n_epoch_batches"=>32,
              "n_epoch"=>75,
              "bestval"=>false,
              "gamma"=>0,
              "self_attention"=>false,
              )