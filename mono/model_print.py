DepthModel(
  (depth_model): DensePredModel(
    (encoder): DinoVisionTransformer(
      (patch_embed): PatchEmbed(
        (proj): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
        (norm): Identity()
      )
      (blocks): ModuleList(
        (0): BlockChunk(
          (0-23): 24 x Block(
            (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (attn): MemEffAttention(
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (ls1): LayerScale()
            (drop_path1): Identity()
            (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ls2): LayerScale()
            (drop_path2): Identity()
          )
        )
      )
      (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (head): Identity()
    )
    (decoder): RAFTDepthNormalDPT5(
      (token2feature): EncoderFeature(
        (read_3): Token2Feature(
          (readoper): Readout(
            (project_patch): Linear(in_features=1024, out_features=1024, bias=True)
            (project_learn): Linear(in_features=5120, out_features=1024, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): Identity()
        )
        (read_2): Token2Feature(
          (readoper): Readout(
            (project_patch): Linear(in_features=1024, out_features=1024, bias=True)
            (project_learn): Linear(in_features=5120, out_features=1024, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): Identity()
        )
        (read_1): Token2Feature(
          (readoper): Readout(
            (project_patch): Linear(in_features=1024, out_features=1024, bias=True)
            (project_learn): Linear(in_features=5120, out_features=1024, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))
        )
        (read_0): Token2Feature(
          (readoper): Readout(
            (project_patch): Linear(in_features=1024, out_features=1024, bias=True)
            (project_learn): Linear(in_features=5120, out_features=1024, bias=False)
            (act): GELU(approximate='none')
          )
          (sample): Sequential(
            (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (decoder_mono): DecoderFeature(
        (upconv_3): FuseBlock(
          (way_trunk): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (out_conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
        (upconv_2): FuseBlock(
          (way_trunk): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (way_branch): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (out_conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        )
        (upconv_1): FuseBlock(
          (way_trunk): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (way_branch): ConvBlock(
            (act): ReLU(inplace=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (out_conv): Conv2d(512, 258, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (depth_regressor): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (normal_predictor): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (3): ReLU(inplace=True)
        (4): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (5): ReLU(inplace=True)
        (6): Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1))
      )
      (context_feature_encoder): ContextFeatureEncoder(
        (outputs04): ModuleList(
          (0-1): 2 x Sequential(
            (0): ResidualBlock(
              (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (relu): ReLU(inplace=True)
              (norm1): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (downsample): Sequential(
                (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
                (1): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (outputs08): ModuleList(
          (0-1): 2 x Sequential(
            (0): ResidualBlock(
              (conv1): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (relu): ReLU(inplace=True)
              (norm1): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (downsample): Sequential(
                (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
                (1): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (outputs16): ModuleList(
          (0-1): 2 x Sequential(
            (0): ResidualBlock(
              (conv1): Conv2d(1024, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (relu): ReLU(inplace=True)
              (norm1): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              (downsample): Sequential(
                (0): Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1))
                (1): LayerNorm2d((128,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (context_zqr_convs): ModuleList(
        (0-2): 3 x Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (update_block): BasicMultiUpdateBlock(
        (gru08): ConvGRU(
          (convz): Conv2d(262, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convr): Conv2d(262, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convq): Conv2d(262, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (gru16): ConvGRU(
          (convz): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convr): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convq): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (gru32): ConvGRU(
          (convz): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convr): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (convq): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (flow_head): FlowHead(
          (conv1d): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2d): Conv2d(128, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv1n): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (conv2n): Conv2d(128, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (relu): ReLU(inplace=True)
        )
        (mask): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
          (2): Conv2d(128, 144, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (relu): ReLU(inplace=True)
    )
  )
)



{'depth_init': torch.Size([1, 6, 52, 52]), 'coords0': torch.Size([1, 6, 52, 52]), 'flow_up': torch.Size([1, 6, 208, 208]), 'predicition': list of 9 tensor with shape [1,1,208,208]