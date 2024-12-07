DepthModel(
  (depth_model): DensePredModel(
    (encoder): DinoVisionTransformer(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(3, 384, kernel_size=(14, 14), stride=(14, 14))
        (norm): Identity()
      )
      (blocks): ModuleList(
        (0): BlockChunk(
          (0-11): 12 x Block(
            (norm1): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
            (attn): MemEffAttention(
              (qkv): Linear(in_features=384, out_features=1152, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=384, out_features=384, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (ls1): LayerScale()
            (drop_path1): Identity()
            (norm2): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=384, out_features=1536, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1536, out_features=384, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ls2): LayerScale()
            (drop_path2): Identity()
          )
        )
      )
      (norm): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
      (head): Identity()
    )
    (decoder): RAFTDepthNormalDPT5(
      (token2feature): EncoderFeature(
        (read_3): Token2Feature(
          (readoper): Readout(
            (project_patch): LoRALinear(in_features=384, out_features=384, bias=True)
            (project_learn): LoRALinear(in_features=1920, out_features=384, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): Identity()
        )
        (read_2): Token2Feature(
          (readoper): Readout(
            (project_patch): LoRALinear(in_features=384, out_features=384, bias=True)
            (project_learn): LoRALinear(in_features=1920, out_features=384, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): Identity()
        )
        (read_1): Token2Feature(
          (readoper): Readout(
            (project_patch): LoRALinear(in_features=384, out_features=384, bias=True)
            (project_learn): LoRALinear(in_features=1920, out_features=384, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): ConvTranspose2dLoRA(384, 192, kernel_size=(2, 2), stride=(2, 2))
        )
        (read_0): Token2Feature(
          (readoper): Readout(
            (project_patch): LoRALinear(in_features=384, out_features=384, bias=True)
            (project_learn): LoRALinear(in_features=1920, out_features=384, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): Sequential(
            (0): Conv2dLoRA(384, 96, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (decoder_mono): DecoderFeature(
        (upconv_3): FuseBlock(
          (way_trunk): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2dLoRA(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2dLoRA(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (out_conv): Conv2dLoRA(384, 384, kernel_size=(1, 1), stride=(1, 1))
        )
        (upconv_2): FuseBlock(
          (way_trunk): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2dLoRA(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2dLoRA(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (way_branch): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2dLoRA(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2dLoRA(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (out_conv): Conv2dLoRA(384, 192, kernel_size=(1, 1), stride=(1, 1))
        )
        (upconv_1): FuseBlock(
          (way_trunk): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2dLoRA(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2dLoRA(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (way_branch): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2dLoRA(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2dLoRA(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (out_conv): Conv2dLoRA(192, 98, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (depth_regressor): Sequential(
        (0): Conv2dLoRA(96, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2dLoRA(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (normal_predictor): Sequential(
        (0): Conv2dLoRA(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2dLoRA(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (3): ReLU(inplace=True)
        (4): Conv2dLoRA(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (5): ReLU(inplace=True)
        (6): Conv2dLoRA(128, 3, kernel_size=(1, 1), stride=(1, 1))
      )
      (context_feature_encoder): ContextFeatureEncoder(
        (outputs04): ModuleList(
          (0-1): 2 x Sequential(
            (0): ResidualBlock(
              (conv1): Conv2dLoRA(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (relu): ReLU(inplace=True)
              (norm1): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (downsample): Sequential(
                (0): Conv2dLoRA(96, 48, kernel_size=(1, 1), stride=(1, 1))
                (1): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (outputs08): ModuleList(
          (0-1): 2 x Sequential(
            (0): ResidualBlock(
              (conv1): Conv2dLoRA(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (relu): ReLU(inplace=True)
              (norm1): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (downsample): Sequential(
                (0): Conv2dLoRA(192, 48, kernel_size=(1, 1), stride=(1, 1))
                (1): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (outputs16): ModuleList(
          (0-1): 2 x Sequential(
            (0): ResidualBlock(
              (conv1): Conv2dLoRA(384, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (relu): ReLU(inplace=True)
              (norm1): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              (downsample): Sequential(
                (0): Conv2dLoRA(384, 48, kernel_size=(1, 1), stride=(1, 1))
                (1): LayerNorm2d((48,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (context_zqr_convs): ModuleList(
        (0-2): 3 x Conv2dLoRA(48, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (update_block): BasicMultiUpdateBlock(
        (gru08): ConvGRU(
          (convz): Conv2dLoRA(102, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convr): Conv2dLoRA(102, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convq): Conv2dLoRA(102, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (gru16): ConvGRU(
          (convz): Conv2dLoRA(144, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convr): Conv2dLoRA(144, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convq): Conv2dLoRA(144, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (gru32): ConvGRU(
          (convz): Conv2dLoRA(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convr): Conv2dLoRA(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convq): Conv2dLoRA(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (flow_head): FlowHead(
          (conv1d): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2d): Conv2dLoRA(48, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv1n): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2n): Conv2dLoRA(48, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (relu): ReLU(inplace=True)
        )
        (mask): Sequential(
          (0): Conv2dLoRA(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2dLoRA(48, 144, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (relu): ReLU(inplace=True)
    )
  )
)