#file	_6	_12	_24	_50	_100	_200
cat<<EOF> data
     1	SS-N	38	38	38	38	38	38
     2	healthCloseIsses12mths0001-hard	47	47	47	47	47	47
     3	pom3c	54	54	54	54	54	54
     4	SS-W	60	60	60	60	60	60
     5	SS-F	65	65	65	65	65	65
     6	Wine_quality	67	67	67	67	67	67
     7	SS-C	68	68	68	68	68	68
     8	pom3b	69	69	69	69	69	69
     9	SS-X	70	70	70	70	70	70
    10	SS-G	72	72	72	72	72	72
    11	Scrum100k	72	72	72	72	72	72
    12	nasa93dem	72	72	72	72	72	72
    13	SQL_AllMeasurements	74	74	74	74	74	74
    14	coc1000	74	74	74	74	74	74
    15	xomo_ground	76	76	76	76	76	76
    16	pom3a	77	77	77	77	77	77
    17	xomo_osp	77	77	77	77	77	77
    18	SS-A	78	78	78	78	78	78
    19	SS-K	78	78	78	78	78	78
    20	xomo_osp2	78	78	78	78	78	78
    21	SS-I	81	81	81	81	81	81
    22	SS-R	81	81	81	81	81	81
    23	SS-O	82	82	82	82	82	82
    24	xomo_flight	83	83	83	83	83	83
    25	SS-B	84	84	84	84	84	84
    26	billing10k	84	84	84	84	84	84
    27	Scrum10k	85	85	85	85	85	85
    28	SS-E	86	86	86	86	86	86
    29	pom3d	86	86	86	86	86	86
    30	auto93	87	87	87	87	87	87
    31	FFM-250-50-0.50-SAT-1	89	89	89	89	89	89
    32	SS-S	89	89	89	89	89	89
    33	FM-500-100-1.00-SAT-1	90	90	90	90	90	90
    34	SS-L	90	90	90	90	90	90
    35	SS-D	91	91	91	91	91	91
    36	FFM-125-25-0.50-SAT-1	92	92	92	92	92	92
    37	SS-V	93	93	93	93	93	93
    38	SS-J	94	94	94	94	94	94
    39	sol-6d-c2-obj1	94	94	94	94	94	94
    40	SS-T	95	95	95	95	95	95
    41	Scrum1k	95	95	95	95	95	95
    42	SS-Q	96	96	96	96	96	96
    43	rs-6d-c3_obj1	96	96	96	96	96	96
    44	wc-6d-c1-obj1	96	96	96	96	96	96
    45	SS-H	97	97	97	97	97	97
    46	SS-U	97	97	97	97	97	97
    47	wc+rs-3d-c4-obj1	97	97	97	97	97	97
    48	wc+wc-3d-c4-obj1	97	97	97	97	97	97
    49	wc+sol-3d-c4-obj1	98	98	98	98	98	98
    50	Apache_AllMeasurements	99	99	99	99	99	99
    51	HSMGP_num	99	99	99	99	99	99
    52	SS-M	99	99	99	99	99	99
    53	SS-P	99	99	99	99	99	99
    54	X264_AllMeasurements	99	99	99	99	99	99
    55	rs-6d-c3_obj2	99	99	99	99	99	99
    56	healthCloseIsses12mths0011-easy	100	100	100	100	100	100
EOF
gnuplot<<EOF
set terminal png size 1000,1000
set output "compare.png"
set key outside
set style data linespoints
set datafile separator whitespace
plot \\
  'data' using 1:3 title '6' with lines, \\
  'data' using 1:4 title '12' with lines, \\
  'data' using 1:5 title '24' with lines, \\
  'data' using 1:6 title '50' with lines, \\
  'data' using 1:7 title '100' with lines, \\
  'data' using 1:8 title '200' with lines
EOF
