import logging


# --------------------------------------------------- #
#                     Logger                          #
# --------------------------------------------------- #
LOG_DESC = "Eyes Water Mark"
# LOG_FORMAT = '[%(asctime)s] - [%(levelname)s] - [%(filename)s] - [%(lineno)d] - [%(threadName)s] - %(message)s'
LOG_FORMAT = '[%(asctime)s] - [%(levelname)s] - [%(threadName)s] - %(message)s'
LOG_LEVEL = logging.DEBUG
LOG_HANDEL_LEVEL = logging.DEBUG


# --------------------------------------------------- #
#                     MySQL                           #
# --------------------------------------------------- #
DB = 'magiccapita'
DB_USER = 'root'
DB_PASSWD = 'root'
DB_HOST = '172.0.0.1'
DB_PORT = 3306
DB_TABLE = 'b_imglist'
# 连接池中空闲连接的初始数量
DB_MIN_CACHED = 10
# 连接池中空闲连接的最大数量
DB_MAX_CACHED = 20
# 共享连接的最大数量
DB_MAX_SHARED = 10
# 创建连接池的最大数量
DB_MAX_CONN = 200
# 超过最大连接数量时候的表现，为True等待连接数量下降，为false直接报错处理
DB_BLOCK = True
# 单个连接的最大重复使用次数
DB_MAX_USAGE = 100
# DB_CHARSET = 'utf8mb4'
DB_CHARSET = 'utf8mb4'
# how connections should be reset when returned to the pool
# False or None to rollback transcations started with begin(),
# True to always issue a rollback for safety's sake
DB_RESET = True
# optional list of SQL commands that may serve to prepare
# the session, e.g. ["set datestyle to ...", "set time zone ..."]
DB_SET_SESSION = None



# --------------------------------------------------- #
#                     Scheduler                       #
# --------------------------------------------------- #
SD_QUERY_INTERVAL = 1 # 轮询时间，分钟
THREAD_POOL_MAX_WORKER = 10
THREAD_NAME_PREFIX = 'eye'


# --------------------------------------------------- #
#                     比较器                          #
# --------------------------------------------------- #
# INIT_SIGMA = 0.5
SIGMA = 1.6
BIAS = 3
INTERVALS = 3
MAX_INTERPOLATION_STEPS = 5
# |D(x^)| < 0.03   0.03极值点太多
DXTHRESHOLD = 5
IMG_BORDER = 5
RATIO = 10

ORI_HIST_BINS = 36
ORI_SIGMA_TIMES = 1.5
ORI_WINDOW_RADIUS = 3.0 * ORI_SIGMA_TIMES
ORI_SMOOTH_TIMES = 2
ORI_PEAK_RATIO = 0.8

DESCR_HIST_BINS = 8
DESCR_WINDOW_WIDTH = 4
DESCR_SCALE_ADJUST = 3
DESCR_MAG_THR = 0.2
INT_DESCR_FCTR = 512.0
FEATURE_ELEMENT_LENGTH = 128

#  最近点与次最近点距离之比,Lowe建议为0.8
NN_SQ_DIST_RATIO_THR=0.8

# --------------------------------------------------- #
#                     水印配置                        #
# --------------------------------------------------- #
WATER_MARK_X = 100
WATER_MARK_Y = 64
# WATER_MARK_FONT = 'Dengb.ttf'
WATER_MARK_FONT = 'msyhbd.ttc' # 经测试，这个字体比较好
WATER_MARK_FONT_SIZE = 14 # 字体大小
WATER_MARK_TEXT_WARP_WIDTH = 10 # 每行文字长度
WATER_MARK_TEXT_LINE_SPACE = 14
EMBEDER_DWT_TIMES = 1 # 小波变换级数，用来断言图片大小，保持这个数即可
EMBEDER_DWT_WINDOW = 2  # 滑动窗口大小 must be 2^n,n >= 1,强烈建议不要改动!!
EMBEDER_DWT_MOD_1 = 32 # 不要随意改动！