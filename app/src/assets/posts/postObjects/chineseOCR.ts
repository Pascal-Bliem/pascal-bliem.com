import Post from "../postModel";

export default new Post(
  // title
  "A data set of handwritten Chinese characters",
  // subtitle
  "Input data for training a deep-learning optical character recognition system",
  // publishDate
  new Date("2020-05-22"),
  // titleImageUrl
  "https://asiasociety.org/sites/default/files/styles/1200w/public/C/calligraphy.jpg",
  // titleImageDescription
  "Let's try to find some data that will help us to decipher Chinese handwriting with ML!",
  // tags
  ["Data Science"],
  // content
  `**TL;DR:** I uploaded data sets with [handwritten](https://www.kaggle.com/pascalbliem/handwritten-chinese-character-hanzi-datasets)  and [handwriting-style font](https://www.kaggle.com/pascalbliem/chinese-characters-from-handwritingstyle-fonts)  Chinese characters on Kaggle, which can be used to train machine learning models for optical character recognition (OCR) on Chinese handwriting. If you're interested in the story behind it, just keep on reading.

I am currently studying Mandarin Chinese, and as you may have guessed, it really isn't the easiest natural language to learn. I remember that when I was learning Indonesian, it only took me around three months to be able to have simple conversations. Now, after about 5 months of studying Chinese for a few hours per week, I am still struggling with the basics. Part of the reason is the pronunciation, which is extremely different form any language I've encountered before. And then there are, of course, the Chinese characters (Hanzi 汉字). There are about 50000 of them, but luckily only about 3000 are used in every day language. Furthermore, the characters used nowadays in mainland China, Singapore, and Malaysia are simplified ones (简体字), whereas Taiwan, Hong Kong, and Macau still use the traditional characters (繁体字). That's definitely enough to utterly confuse me already, and since they're pictograms that don't convey any phonetic information, it is tricky to just find them on a keyboard to type them into a translation service to look up the meaning. Google [Translate](https://translate.google.com/) has a camera-based optical character recognition (OCR) function and Chinese dictionary apps like [Pleco](https://www.pleco.com/) offer OCR services as well, but they seem to be specialized on printed fonts and do not perform as well on handwritten characters. So what am I going to do when I want to decipher the sloppily written text on a sticky note that my friend left for on the fridge for me? I may try to build my own OCR system focused on handwriting, but first of all, I would need to find some data.

### What is out there already 

Considering that OCR has been a typical machine learning problem for decades already, the internet isn't exactly flooded with Chinese character data sets, especially when it comes to handwritten ones. And when it comes to sources in English language; due to China's great firewall, purely Chinese digital ecosystems have developed in which it's hard to find even a trace of English. I tried to searching through some [Baidu](http://www.baidu.com/) file sharing platform with my lousy Chinese skills and the help of Google Translate, but couldn't even register an account to download things because it seemed to require a Chinese mobile phone number. So, I thought, let's search the places a data scientist would usually go to: [Github](https://github.com/) and [Kaggle](https://www.kaggle.com/). There are some search results popping up when screening Github for "Chinese OCR" or "Chinese text recognition", but unsurprisingly, most of them (like [this](https://github.com/Wang-Shuo/Chinese-Text-Detection-and-Recognition) one, [this](https://github.com/wycm/xuexin-ocr) one, or [this](https://github.com/chineseocr/chineseocr) one) are written in Chinese (I should really put more effort in getting more fluent in it quickly). Most of them didn't seem to focus on handwritten characters anyway. One of the few things I found on Github in English was an amazing *[Chinese Text in the Wild](https://ctwdataset.github.io/)* data set with annotated images of occurrences of Chinese text from the perspective of a car-mounted camera (see image below); which unfortunately isn't about handwriting either.

![An example of Chinese Text in the Wild form a car's perspective.](https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/hanzi-dataset/textinthewild.png)

Kaggle was going to be my next stop. The only existing [data set](https://www.kaggle.com/dylanli/chinesecharacter) I found there was a collection of Chinese fonts files, which could be used for generating images of characters from the fonts. It only contained 17 fonts though, meaning only 17 images per character (without image augmentation), plus maybe 9 more which I could find on [Google Fonts](https://fonts.google.com/?subset=chinese-simplified). That hardly seemed enough. But the idea wasn't a bad one: generating images from fonts. I came across a [blog post](https://blog.usejournal.com/making-of-a-chinese-characters-dataset-92d4065cc7cc) of Peter Burkimsher, who took a very ambitious take on this idea. He did not just care about maybe the 3000 most commonly used characters or even the 7330 characters in the GB2312 encoding, but he created 15 million image files of 52,835 characters. I downloaded the compressed archive which he kindly provided, but I had to give up before I could even finish to unpack it because it blew up my hard drive. Again, most of the fonts didn't resemble handwriting anyway, but his post pointed me towards another great resource: a website called [chinesefontdesign.com](https://chinesefontdesign.com/).

### Creating Chinese character images form fonts files

I browsed said website for handwriting-style fonts. Some manual selection was necessary here because some of the fonts were only for traditional characters (I was looking for the simplified ones only) and others labeled as handwriting-style actually looked nothing like human handwriting. I managed to find around 120 fonts files which I could use for generating images. Python's \`PIL\` package has some useful functionality for doing so. Here's a little code snipped that I used:
\`\`\`python
# libraries for image processing
from PIL import Image, ImageDraw, ImageFont, ImageChops
import cv2
# libraries for I/O
import os
import glob

# some common simplified Chinese characters and alphanumerics
charset = '0123456789QWERTYUIOPASDFGHJKLZXCVBNMqwertzuiopasdfghjklyxcvbnm.:;-+!$#%&@“”《》的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事成方多经么去法学如都同现当没动面起看定天分还进好小部其些主样理心她本前开但因只从想实日军者意无力它与长把机十民第公此已工使情明性知全三又关点正业外将两高间由问很最重并物手应战向头文体政美相见被利什二等产或新己制身果加西斯月话合回特代内信表化老给世位次度门任常先海通教儿原东声提立及比员解水名真论处走义各入几口认条平系气题活尔更别打女变四神总何电数安少报才结反受目太量再感建务做接必场件计管期市直德资命山金指克许统区保至队形社便空决治展马科司五基眼书非则听白却界达光放强即像难且权思王象完设式色路记南品住告类求据程北边死张该交规万取拉格望觉术领共确传师观清今切院让识候带导争运笑飞风步改收根干造言联持组每济车亲极林服快办议往元英士证近失转夫令准布始怎呢存未远叫台单影具罗字爱击流备兵连调深商算质团集百需价花党华城石级整府离况亚请技际约示复病息究线似官火断精满支视消越器容照须九增研写称企八功吗包片史委乎查轻易早曾除农找装广显吧阿李标谈吃图念六引历首医局突专费号尽另周较注语仅考落青随选列武红响虽推势参希古众构房半节土投某案黑维革划敌致陈律足态护七兴派孩验责营星够章音跟志底站严巴例防族供效续施留讲型料终答紧黄绝奇察母京段依批群项故按河米围江织害斗双境客纪采举杀攻父苏密低朝友诉止细愿千值仍男钱破网热助倒育属坐帝限船脸职速刻乐否刚威毛状率甚独球般普怕弹校苦创假久错承印晚兰试股拿脑预谁益阳若哪微尼继送急血惊伤素药适波夜省初喜卫源食险待述陆习置居劳财环排福纳欢雷警获模充负云停木游龙树疑层冷洲冲射略范竟句室异激汉村哈策演简卡罪判担州静退既衣您宗积余痛检差富灵协角占配征修皮挥胜降阶审沉坚善妈刘读啊超免压银买皇养伊怀执副乱抗犯追帮宣佛岁航优怪香著田铁控税左右份穿艺背阵草脚概恶块顿敢守酒岛托央户烈洋哥索胡款靠评版宝座释景顾弟登货互付伯慢欧换闻危忙核暗姐介坏讨丽良序升监临亮露永呼味野架域沙掉括舰鱼杂误湾吉减编楚肯测败屋跑梦散温困剑渐封救贵枪缺楼县尚毫移娘朋画班智亦耳恩短掌恐遗固席松秘谢鲁遇康虑幸均销钟诗藏赶剧票损忽巨炮旧端探湖录叶春乡附吸予礼港雨呀板庭妇归睛饭额含顺输摇招婚脱补谓督毒油疗旅泽材灭逐莫笔亡鲜词圣择寻厂睡博勒烟授诺伦岸奥唐卖俄炸载洛健堂旁宫喝借君禁阴园谋宋避抓荣姑孙逃牙束跳顶玉镇雪午练迫爷篇肉嘴馆遍凡础洞卷坦牛宁纸诸训私庄祖丝翻暴森塔默握戏隐熟骨访弱蒙歌店鬼软典欲萨伙遭盘爸扩盖弄雄稳忘亿刺拥徒姆杨齐赛趣曲刀床迎冰虚玩析窗醒妻透购替塞努休虎扬途侵刑绿兄迅套贸毕唯谷轮库迹尤竞街促延震弃甲伟麻川申缓潜闪售灯针哲络抵朱埃抱鼓植纯夏忍页杰筑折郑贝尊吴秀混臣雅振染盛怒舞圆搞狂措姓残秋培迷诚宽宇猛摆梅毁伸摩盟末乃悲拍丁赵硬麦蒋操耶阻订彩抽赞魔纷沿喊违妹浪汇币丰蓝殊献桌啦瓦莱援译夺汽烧距裁偏符勇触课敬哭懂墙袭召罚侠厅拜巧侧韩冒债曼融惯享戴童犹乘挂奖绍厚纵障讯涉彻刊丈爆乌役描洗玛患妙镜唱烦签仙彼弗症仿倾牌陷鸟轰咱菜闭奋庆撤泪茶疾缘播朗杜奶季丹狗尾仪偷奔珠虫驻孔宜艾桥淡翼恨繁寒伴叹旦愈潮粮缩罢聚径恰挑袋灰捕徐珍幕映裂泰隔启尖忠累炎暂估泛荒偿横拒瑞忆孤鼻闹羊呆厉衡胞零穷舍码赫婆魂灾洪腿胆津俗辩胸晓劲贫仁偶辑邦恢赖圈摸仰润堆碰艇稍迟辆废净凶署壁御奉旋冬矿抬蛋晨伏吹鸡倍糊秦盾杯租骑乏隆诊奴摄丧污渡旗甘耐凭扎抢绪粗肩梁幻菲皆碎宙叔岩荡综爬荷悉蒂返井壮薄悄扫敏碍殖详迪矛霍允幅撒剩凯颗骂赏液番箱贴漫酸郎腰舒眉忧浮辛恋餐吓挺励辞艘键伍峰尺昨黎辈贯侦滑券崇扰宪绕趋慈乔阅汗枝拖墨胁插箭腊粉泥氏彭拔骗凤慧媒佩愤扑龄驱惜豪掩兼跃尸肃帕驶堡届欣惠册储飘桑闲惨洁踪勃宾频仇磨递邪撞拟滚奏巡颜剂绩贡疯坡瞧截燃焦殿伪柳锁逼颇昏劝呈搜勤戒驾漂饮曹朵仔柔俩孟腐幼践籍牧凉牲佳娜浓芳稿竹腹跌逻垂遵脉貌柏狱猜怜惑陶兽帐饰贷昌叙躺钢沟寄扶铺邓寿惧询汤盗肥尝匆辉奈扣廷澳嘛董迁凝慰厌脏腾幽怨鞋丢埋泉涌辖躲晋紫艰魏吾慌祝邮吐狠鉴曰械咬邻赤挤弯椅陪割揭韦悟聪雾锋梯猫祥阔誉筹丛牵鸣沈阁穆屈旨袖猎臂蛇贺柱抛鼠瑟戈牢逊迈欺吨琴衰瓶恼燕仲诱狼池疼卢仗冠粒遥吕玄尘冯抚浅敦纠钻晶岂峡苍喷耗凌敲菌赔涂粹扁亏寂煤熊恭湿循暖糖赋抑秩帽哀宿踏烂袁侯抖夹昆肝擦猪炼恒慎搬纽纹玻渔磁铜齿跨押怖漠疲叛遣兹祭醉拳弥斜档稀捷肤疫肿豆削岗晃吞宏癌肚隶履涨耀扭坛拨沃绘伐堪仆郭牺歼墓雇廉契拼惩捉覆刷劫嫌瓜歇雕闷乳串娃缴唤赢莲霸桃妥瘦搭赴岳嘉舱俊址庞耕锐缝悔邀玲惟斥宅添挖呵讼氧浩羽斤酷掠妖祸侍乙妨贪挣汪尿莉悬唇翰仓轨枚盐览傅帅庙芬屏寺胖璃愚滴疏萧姿颤丑劣柯寸扔盯辱匹俱辨饿蜂哦腔郁溃谨糟葛苗肠忌溜鸿爵鹏鹰笼丘桂滋聊挡纲肌茨壳痕碗穴膀卓贤卧膜毅锦欠哩函茫昂薛皱夸豫胃舌剥傲拾窝睁携陵哼棉晴铃填饲渴吻扮逆脆喘罩卜炉柴愉绳胎蓄眠竭喂傻慕浑奸扇柜悦拦诞饱乾泡贼亭夕爹酬儒姻卵氛泄杆挨僧蜜吟猩遂狭肖甜霞驳裕顽於摘矮秒卿畜咽披辅勾盆疆赌塑畏吵囊嗯泊肺骤缠冈羞瞪吊贾漏斑涛悠鹿俘锡卑葬铭滩嫁催璇翅盒蛮矣潘歧赐鲍锅廊拆灌勉盲宰佐啥胀扯禧辽抹筒棋裤唉朴咐孕誓喉妄拘链驰栏逝窃艳臭纤玑棵趁匠盈翁愁瞬婴孝颈倘浙谅蔽畅赠妮莎尉冻跪闯葡後厨鸭颠遮谊圳吁仑辟瘤嫂陀框谭亨钦庸歉芝吼甫衫摊宴嘱衷娇陕矩浦讶耸裸碧摧薪淋耻胶屠鹅饥盼脖虹翠崩账萍逢赚撑翔倡绵猴枯巫昭怔渊凑溪蠢禅阐旺寓藤匪伞碑挪琼脂谎慨菩萄狮掘抄岭晕逮砍掏狄晰罕挽脾舟痴蔡剪脊弓懒叉拐喃僚捐姊骚拓歪粘柄坑陌窄湘兆崖骄刹鞭芒筋聘钩棍嚷腺弦焰耍俯厘愣厦恳饶钉寡憾摔叠惹喻谱愧煌徽溶坠煞巾滥洒堵瓷咒姨棒郡浴媚稣淮哎屁漆淫巢吩撰啸滞玫硕钓蝶膝姚茂躯吏猿寨恕渠戚辰舶颁惶狐讽笨袍嘲啡泼衔倦涵雀旬僵撕肢垄夷逸茅侨舆窑涅蒲谦杭噢弊勋刮郊凄捧浸砖鼎篮蒸饼亩肾陡爪兔殷贞荐哑炭坟眨搏咳拢舅昧擅爽咖搁禄雌哨巩绢螺裹昔轩谬谍龟媳姜瞎冤鸦蓬巷琳栽沾诈斋瞒彪厄咨纺罐桶壤糕颂膨谐垒咕隙辣绑宠嘿兑霉挫稽辐乞纱裙嘻哇绣杖塘衍轴攀膊譬斌祈踢肆坎轿棚泣屡躁邱凰溢椎砸趟帘帆栖窜丸斩堤塌贩厢掀喀乖谜捏阎滨虏匙芦苹卸沼钥株祷剖熙哗劈怯棠胳桩瑰娱娶沫嗓蹲焚淘嫩韵衬匈钧竖峻豹捞菊鄙魄兜哄颖镑屑蚁壶怡渗秃迦旱哟咸焉谴宛稻铸锻伽詹毙恍贬烛骇芯汁桓坊驴朽靖佣汝碌迄冀荆崔雁绅珊榜诵傍彦醇笛禽勿娟瞄幢寇睹贿踩霆呜拱妃蔑谕缚诡篷淹腕煮倩卒勘馨逗甸贱炒灿敞蜡囚栗辜垫妒魁谣寞蜀甩涯枕丐泳奎泌逾叮黛燥掷藉枢憎鲸弘倚侮藩拂鹤蚀浆芙垃烤晒霜剿蕴圾绸屿氢驼妆捆铅逛淑榴丙痒钞蹄犬躬昼藻蛛褐颊奠募耽蹈陋侣魅岚侄虐堕陛莹荫狡阀绞膏垮茎缅喇绒搅凳梭丫姬诏钮棺耿缔懈嫉灶匀嗣鸽澡凿纬沸畴刃遏烁嗅叭熬瞥骸奢拙栋毯桐砂莽泻坪梳杉晤稚蔬蝇捣顷麽尴镖诧尬硫嚼羡沦沪旷彬芽狸冥碳咧惕暑咯萝汹腥窥俺潭崎麟捡拯厥澄萎哉涡滔暇溯鳞酿茵愕瞅暮衙诫斧兮焕棕佑嘶妓喧蓉删樱伺嗡娥梢坝蚕敷澜杏绥冶庇挠搂倏聂婉噪稼鳍菱盏匿吱寝揽髓秉哺矢啪帜邵嗽挟缸揉腻驯缆晌瘫贮觅朦僻隋蔓咋嵌虔畔琐碟涩胧嘟蹦冢浏裔襟叨诀旭虾簿啤擒枣嘎苑牟呕骆凸熄兀喔裳凹赎屯膛浇灼裘砰棘橡碱聋姥瑜毋娅沮萌俏黯撇粟粪尹苟癫蚂禹廖俭帖煎缕窦簇棱叩呐瑶墅莺烫蛙歹伶葱哮眩坤廓讳啼乍瓣矫跋枉梗厕琢讥釉窟敛轼庐胚呻绰扼懿炯竿慷虞锤栓桨蚊磅孽惭戳禀鄂馈垣溅咚钙礁彰豁眯磷雯墟迂瞻颅琉悼蝴拣渺眷悯汰慑婶斐嘘镶炕宦趴绷窘襄珀嚣拚酌浊毓撼嗜扛峭磕翘槽淌栅颓熏瑛颐忖'

# read in the fonts from the provided files
font_paths = glob.glob("./font_files/*")
fonts = [ImageFont.truetype(p, 75, 0) for p in font_paths]

# Here are a few functions for creating an image of a character from fonts:

# add some blur to avoid unrealistically sharp edges on characters
def blur_image(im, kernel=(0,0), std=1):
    return Image.fromarray(cv2.GaussianBlur(np.array(im), kernel, std))

# trim excessive whitespace around the character
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im

# create an image from a font file   
def setup_char_image(char, font, shape=(75,75)):
    image = np.ones(shape=shape, dtype=np.uint8)
    x = Image.fromarray(image)
    draw = ImageDraw.Draw(x)
    draw.text((0,0),char,(0),font=font)
    p = ((255.0 / np.array(x).max() * (np.array(x) - np.array(x).min()))
        .astype(np.uint8))
    return Image.fromarray(p)

# create the images using the functions above
def create_char_image(charset,fonts):
    # the images will be stored in folders labeled as the respective
    # char - some chars cannot be used in folder names in some OS
    forbidden_chars = {"<": "less_than",
                       ">": "greater_than",
                       ":": "colon",
                       '"': "double_quote",
                       "/": "forward_slash",
                       "\\\\": "backslash",
                       "|": "vertical_bar",
                       "?": "question_mark",
                       "*": "asterisk",
                       ".": "full_stop"}
    
    # iterate over all chars and fonts
    for char in charset:
        for font in fonts:
            dir_name = (char 
                        if not char in forbidden_chars.keys() 
                        else forbidden_chars[char])
            file_name = str(font)[38:-1]
            
            save_path = "data/" + dir_name
            if not os.path.exists(save_path):
                os.makedirs(save_path)
    
            trim(
            blur_image(
            setup_char_image(char,font)
            )).resize((75,75)
            ).save(save_path + "/" + file_name + ".png")


if __name__ == "__main__":
    # create images of chinese chars in charset form the provided fonts files
    create_char_image(charset,fonts)
\`\`\`
I have provided this script along with the font files as a [public data set](https://www.kaggle.com/pascalbliem/chinese-characters-from-handwritingstyle-fonts) on Kaggle, feel free to use it yourself. Let's now have a look at the images this method produced.

<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/hanzi-dataset/from_fonts.png" alt="Images of the Chinese character 两, generated from font files." style="width: 80%;">

The results are actually not bad. Some of the characters do really look quite a bit like handwriting. However, even with data augmentation, I knew about 120 images per character were not going to be enough if I want to classify thousands of different characters. 

### Getting real handwritten data

I kept searching for data sets of actual handwritten characters and eventually found some resources. One of them were two datasets from the Harbin Institute of Technology called [HIT-MW](https://sites.google.com/site/hitmwdb/) and [HIT-OR3C](http://www.iapr-tc11.org/mediawiki/index.php?title=Harbin_Institute_of_Technology_Opening_Recognition_Corpus_for_Chinese_Characters_(HIT-OR3C)). Another great recourse I cam across were the [CASIA Online and Offline Chinese Handwriting Databases](http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html) created by the Chinese National Laboratory of Pattern Recognition (NLPR) and the Institute of Automation of Chinese Academy of Sciences (CASIA). They had collected about 3.9 million writing samples of 1020 writers, overall 7185 Chinese characters and 171 symbols. This sounded pretty impressive to me and I saw that [some](https://github.com/soloice/Chinese-Character-Recognition) [other](https://pdfs.semanticscholar.org/4941/aed85462968e9918110b4ba740c56030fd23.pdf) works had successfully used this dataset, so I decided to go with this one as well. The data files are available for [download](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html) on their website; however, not exactly *ready to use*.

Instead of image files, the authors provided their data in the form of binary files with some custom encoding:

<div class="post-page-table">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
    <table border="1" class="dataframe">
		<tr>
			<td width="64">Item</td>
			<td width="76">Type</td>
			<td width="77">Length</td>
			<td width="111">Instance</td>
			<td width="162">Comment</td>
		</tr>
		<tr>
			<td>Sample size</td>
			<td>unsigned int</td>
			<td>4B</td>
			<td>&nbsp;</td>
			<td>Number of bytes for one sample (byte count to next sample)</td>
		</tr>
		<tr>
			<td>Tag code (GB)</td>
			<td>char</td>
			<td>2B</td>
			<td>&quot;啊&quot;=0xb0a1 Stored as 0xa1b0</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>Width</td>
			<td>unsigned short</td>
			<td>2B</td>
			<td>&nbsp;</td>
			<td>Number of pixels in a row</td>
		</tr>
		<tr>
			<td>Height</td>
			<td>unsigned short</td>
			<td>2B</td>
			<td>&nbsp;</td>
			<td>Number of rows</td>
		</tr>
		<tr>
			<td>Bitmap</td>
			<td>unsigned char</td>
			<td>Width*Height bytes</td>
			<td>&nbsp;</td>
			<td>Stored row by row</td>
		</tr>
    </table>
</div>

It took me a while to figure out the proper way to handle binary files with Python. The \`struct\` package seems to have everything needed to customly decode everything byte by byte (you can find a list of all \`struct\` format characters [here](https://docs.python.org/3/library/struct.html#format-characters)) and with \`PIL\`, we can turn it into image files:

\`\`\`python
# handling binary files
import struct
# image processing
from PIL import Image, ImageDraw
import numpy as np
# some I/O functionality
import glob
import os

# path to folder containing the unzipped binary files
data_folder = "Gnt1.0TrainPart1"
# path of the folder the images should be saved in
train_test = "Train"

# iterate over all the unpacked .gnt binary files
for path in glob.glob(data_folder + "/*.gnt"):
    
    filesize = os.path.getsize(path)
    
    with open(path, "rb") as file:
        content = file.read()
    
    # while the counter is smaller than the size of the file, keep iterating
    counter = 0
    while counter < filesize:
        
        # size in bytes of one character sample
        sample_size = struct.unpack("I",content[counter:counter+4])[0]
        
        # unpack th two tag codes that represent the character label
        # and merge them together (ignoring NULL bytes b'\\x00')
        tag_code1 = struct.unpack("cc",content[counter+4:counter+6])[0]
        tag_code2 = struct.unpack("cc",content[counter+4:counter+6])[1]
        tag_code = ((tag_code1 + tag_code2).decode("GBK") 
                     if tag_code2 != b'\\x00' 
                     else (tag_code1).decode("GBK"))
        
        # the images will be stored in folders labeled as the respective
        # character - some chars cannot be used in folder names in some OS
        forbidden_chars = {"<": "less_than",
                           ">": "greater_than",
                           ":": "colon",
                           '"': "double_quote",
                           "/": "forward_slash",
                           "\\\\": "backslash",
                           "|": "vertical_bar",
                           "?": "question_mark",
                           "*": "asterisk",
                           ".": "full_stop"}
    
        if tag_code in forbidden_chars.keys():
            tag_code = forbidden_chars[tag_code]
        
        # unpack width and hight of the writing sample
        width = struct.unpack("H",content[counter+6:counter+8])[0]
        height = struct.unpack("H",content[counter+8:counter+10])[0]
        area = width * height
        
        # unpack the bitmap that represents the image of the writing sample
        bitmap = (np.array(struct.unpack("B" * area,
                                         content[counter+10:counter+10+area]))
                  .reshape(height,width))
        
        bitmap = np.where(bitmap!=255,
                          ((255.0 / bitmap.max() * (bitmap - bitmap.min())).astype(np.uint8)),
                          bitmap).astype(np.uint8)
        
        # create an image object from the bitmap
        image = Image.fromarray(bitmap)
        ImageDraw.Draw(image)
        
        # save the image in a folder labeled as the corresponding character
        save_path = train_test + f"/{tag_code}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = str(len(glob.glob(save_path + "/*"))+1) + ".png"
        image.save(save_path + "/" + file_name)
        
        # increment the byte counter 
        counter += sample_size

\`\`\`
A sample of the resulting character images can be seen in the figure below. It is quite remarkable how different the handwriting styles of different writers can be. For some of the images, I couldn't have said with certainty what character they display. This may be even more challenging for a machine learning algorithm and highlights why many currently available OCR apps fail at recognizing handwriting.

<img src="https://pb-data-blogposts.s3.eu-central-1.amazonaws.com/hanzi-dataset/from_hand.png" alt="Images of the Chinese character 两, written by hand by different writers." style="width: 80%;">

Downloading, unpacking, and processing all this data took quite a while. If you want to skip the effort and work with image files directly, I have put them on Kaggle as a [public dataset](https://www.kaggle.com/pascalbliem/handwritten-chinese-character-hanzi-datasets). Feel free to use them and tell me about your results.

### What next?

I originally had the intention to build an OCR app based on this data set. There are plenty of OCR apps for printed characters available and I also found a [paper](https://pdfs.semanticscholar.org/4941/aed85462968e9918110b4ba740c56030fd23.pdf) from Stanford student in which they trained a deep convolutional neural network on a subsection of this dataset and got great accuracy (around 95%) even with more than 3000 classes. So, I started off quite optimistic. Unfortunately, I don't have a GPU and I realized that this task (classifying thousands of classes on millions of images) was absolutely intractable on my hardware, even if I had my backup laptop running for weeks. So, were could I get a GPU from without spending money? Since Kaggle has introduced its 30 hour limit, there is not much room for experimentation on this platform anymore. I tried to use [Google Colab](https://colab.research.google.com/), which offers free GPU time with some limitation, in combination with my Google Drive for persisting model checkpoints, but with little success. Somehow, Colab's file system couldn't handle unpacking millions of files and threw some errors all the time or leaving me with empty files which, of course, crashed TensorFlow when it tried to decode such "images". Buying GPU time from cloud providers isn't exactly cheap either. I'm mostly active on AWS, where the cheapest GPU instance type costs around 1.3 USD per hour. I decided that I wasn't willing to spend a non-negligible amount of money on this project *just for fun* and ended it here.

If you have some powerful GPUs in your possession and an interest in Chinese OCR, why don't you try to pick up the challenge where I left of and tell me how it went? I'd be really interested in seeing how well a handwriting OCR app would work.

`
);
