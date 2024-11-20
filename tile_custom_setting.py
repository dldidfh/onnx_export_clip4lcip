L_TILE_SIZE = 672
M_TILE_SIZE = 540
S_TILE_SIZE = 448

ROI = {
    "L":[
        [0, L_TILE_SIZE, 0, L_TILE_SIZE],
        [0, L_TILE_SIZE, 504, 504+L_TILE_SIZE],
        [0, L_TILE_SIZE, 1008, 1008+L_TILE_SIZE ],
        [408, 408 + L_TILE_SIZE, 0, L_TILE_SIZE],
        [408, 408 + L_TILE_SIZE, 504, 504+L_TILE_SIZE],
        [408, 408 + L_TILE_SIZE, 1008, 1008+L_TILE_SIZE],
        [0, L_TILE_SIZE, 1248, 1248+L_TILE_SIZE ],
        [408, 408 + L_TILE_SIZE, 1248, 1248+L_TILE_SIZE ],
    ],
    "M": [
        [0, M_TILE_SIZE, 0, M_TILE_SIZE],        
        [0, M_TILE_SIZE, 420, 420 + M_TILE_SIZE],
        [0, M_TILE_SIZE, 840, 840 + M_TILE_SIZE], 
        [0, M_TILE_SIZE, 1260, 1260 + M_TILE_SIZE], 

        [420, 420 + M_TILE_SIZE, 0, M_TILE_SIZE],        
        [420, 420 + M_TILE_SIZE, 420, 420 + M_TILE_SIZE],
        [420, 420 + M_TILE_SIZE, 840, 840 + M_TILE_SIZE], 
        [420, 420 + M_TILE_SIZE, 1260, 1260 + M_TILE_SIZE], 

        [520, 520 + M_TILE_SIZE, 0, M_TILE_SIZE],        
        [520, 520 + M_TILE_SIZE, 420, 420 + M_TILE_SIZE],
        [520, 520 + M_TILE_SIZE, 840, 840 + M_TILE_SIZE], 
        [520, 520 + M_TILE_SIZE, 1260, 1260 + M_TILE_SIZE], 

        [0, 0 + M_TILE_SIZE, 1360, 1360 + M_TILE_SIZE],        
        [420, 420 + M_TILE_SIZE, 1360, 1360 + M_TILE_SIZE],

        [520, 520 + M_TILE_SIZE, 1360, 1360 + M_TILE_SIZE],
    ]
}