import gradio as gr
import json
import os
import pandas as pd
import re

try:
    from modules import script_callbacks,scripts
    import modules
    import modules.shared as shared
    tiao=False
except:
    tiao=True

# 第三版改进计划如下：
# √ tag后加逗号选项改进
# √ tag逗号和tag括号混用记得可以隐藏
# √ 设置记忆功能item
# √ 一些选项的填空选择主页
# √ 隐藏和搜索自定义关键词(隐藏关键词为单个连续)
# √ 滑块步长的设置
# √ 更新embedding按钮
# √ 滑块最大值最小值的设置
# 将check_to_sub提交后得到check一次性清空是否？
# tag分类再细化
# 元素法典正反tag分类再细化
# 括号混合优化之将多数赋予1权值
# name作为自定义单个咒语的中文
# √ 更新自定义咒语中文与num和index
# √ 简化代码，采用lamdba进行简化
# 简化代码，采用单个相互联系法
# 选择是否同步隐藏和搜索关键词在元素法典和魔法碎片
# 全选与反选自定义
# √ enter进行搜索
# 魔法混合时的参数调节选项
# 增加冒号后第二个参数
# √ 括号混合优化，混合后根据是否隐藏括号混合选项进行调节
# √ 英译汉翻译元素法典中的tag
# ○ 元素法典第二点五卷整理
# √ 元素法典关键词分类识别check
# 关键词分组与分类
# √ 正面tag与反面tag的主页分离，即分开两个阵营(设两个可以互动的输出框)
# 添加组合tag中的那些子tag
# 修改tag的api替代勾选选项
# √ 元素法典注释一栏优化
# √ 去掉dict的\n符号为空格
# √ 一直全部大括号或小括号
# √ seach函数优化，关键词连续前置，检查tags与chin
# 禁术名字按sorted排列，数字前面补0
# √ 元素法典单条咒语分类添加，增加一个check并全部选中，采用in，方便@分隔形式
# √ gradio库原格式style.css
# √ pt文件遍历添加
# √ 遗失魔法碎片的排序快速搜索
# √ 删除单个tag
# √ 直接跳转填充正面tag和反面tag
# 一键导入通用配置
# √ from to when语法(分步描绘)，将:作为与逗号等同的参数
# 自然语言咏唱，实现英文整句翻译
# √ emoij大全
# 清除权重为零
# √ 占位符和强调咏唱AND，当成一个tag进行，在自定义处
# 自定义分类文件tag
# √ 最后一个永久去逗号
# √ 融合吟唱，|作为与逗号等同的参数
# √ 纠正元素法典里面的融合吟唱改错|2@0@3
# 将关键词自定义添加提前
# 元素法典咒语变式
# 将step等参数一键传递设置好元素法典传递
# √ 更新的版本问题
# √ 将保存的item简化为数字，使其体积更小
# enter时生效设置魔法碎片
# 遗失魔法碎片之最小值从第几个结果开始显示
# 自定义占位符

# 一些未修好的bug：
# 关于radio的value值以index形式保存

# bug:
# √ 保存特殊符号为咒语时记得去掉$，否则容易重复，记得check_to_sub时候将time加上去，time可以设为Var参数

path="extensions/maple-from-fall-and-flower/scripts"
if not os.path.exists(path+"/search.json"):
    path="\\".join(__file__.split("\\")[:-1])
maxmax=770596

with open(path+"/search.json") as search:
    search = json.load(search)
with open(path+"/tags.json") as tags:
    tags = json.load(tags)
with open(path+"/storage.json", "w", encoding="utf-8") as storage:
    storage.write("{}")
try:
    with open(path+"/magic.json") as magic:
        magic=json.load(magic)
    magicperfect=True
except:
    magicperfect=False
try:
    with open(path+"/dict.json") as ddict:
        ddict=json.load(ddict)
    with open (path+"/find.json") as find:
        find=json.load(find)
    dictperfect=True
except:
    dictperfect=False

if not os.path.exists(path+"/item.json"):
    with open (path+"/item.json","w",encoding="utf-8") as ite:
        ite.write("{}")

choli=["常用 优化Tag","常用 其他Tag","常用 R18Tag","环境 朝朝暮暮","环境 日月星辰","环境 天涯海角","风格","非emoij的人物","角色","头发&发饰 长度","头发&发饰 颜色","头发&发饰 发型","头发&发型 辫子","头发&发型 刘海/其他","头发&发型 发饰","五官&表情 常用","五官&表情 R18","眼睛 颜色","眼睛 状态","眼睛 其他","身体 胸","身体 R18","服装 衣服","服装 R18","袜子&腿饰 袜子","袜子&腿饰 长筒袜","袜子&腿饰 连裤袜","袜子&腿饰 腿饰&组合","袜子&腿饰 裤袜","袜子&腿饰 R18","鞋 鞋子","装饰 装饰","动作 动作","动作 头发相关","动作 R18","Emoij😊 表情","Emoij😊 人物","Emoij😊 手势","Emoij😊 日常","Emoij😊 动物","Emoij😊 植物","Emoij😊 自然","Emoij😊 食物","R18 ","人体","姿势","发型","表情","眼睛","衣服","饰品","袜子","风格(画质)","环境","背景","物品"]
emoji=['Emoji 物品 💊医疗7', 'Emoji 旅行和地点 ⛽陆路交通50', 'Emoji 食物和饮料 🍅水果19', 'Emoji 旗帜 🚩旗子8', 'Emoji 笑脸和情感 😛吐舌脸6', 'Emoji 符号 0️⃣键帽13', 'Emoji 动物和自然 🐍爬行动物8', 'Emoji 人类和身体 👃部分身体18', 'Emoji 符号 ♀️性别3', 'Emoji 动物和自然 🌹花朵11', 'Emoji 旅行和地点 ☂️天空和天气47', 'Emoji 活动 🎯游戏24', 'Emoji 肤色和发型 🏽肤色5', 'Emoji 物品 📒图书与纸张17', 'Emoji 笑脸和情感 😄笑脸14', 'Emoji 符号 ⚪️几何34', 'Emoji 笑脸和情感 😸猫咪脸9', 'Emoji 动物和自然 🐓鸟类18', 'Emoji 人类和身体 👌几根手指9', 'Emoji 物品 �居居家25','Emoji 人类和身体 👈一根手指7', 'Emoji 符号 ✖数学符号6', 'Emoji 活动 🏅奖牌6', 'Emoji 活动 🏀运动27', 'Emoji 笑脸和情感 🙈猴子脸3', 'Emoji 人类和身体 🚴人物运动43', 'Emoji 物品 🖱️电脑14', 'Emoji 笑脸和情感 😠消极脸8', 'Emoji 动物和自然 🌴其他植物16', 'Emoji 食物和饮料 🦀 海产5', 'Emoji 旅行和地点 ⌚时间31', 'Emoji 动物和自然 🐟海洋动物11', 'Emoji 笑脸和情感 ❤爱心22', 'Emoji 物品 📞电话6', 'Emoji 人类和身体 🏃人物 活动39', 'Emoji 食物和饮料 ☕饮料20', 'Emoji 笑脸和情感 🤔带手脸7', 'Emoji 符号 ☑️其他符号21', 'Emoji 物品 📢声音9', 'Emoji 人类和身体 🛌人 物休息5', 'Emoji 人类和身体 👣人物标记5', 'Emoji 笑脸和情感 😍表情脸9', 'Emoji 动物和自然 🐸两栖动物1', 'Emoji 旗帜 🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f地区旗3', 'Emoji 笑脸和情感 😎眼镜脸3', 'Emoji 符号 🅰字符39', 'Emoji 符号 ☪️ 宗教12', 'Emoji 物品 💲金钱10', 'Emoji 人类和身体 👨\u200d🍳人物角色82', 'Emoji 旅行和地点 🌍地图7', 'Emoji 人类和身体 👦人物28', 'Emoji 食物和饮料 🍦甜食14', 'Emoji 物品 🚬其他物品9', 'Emoji 旅行和地点 🏗️建筑27', 'Emoji 物品 ⛏️工具25', 'Emoji 物品 🔏锁6', 'Emoji 活动 🎨艺术和工艺7', 'Em oji 笑脸和情感 💋情感14', 'Emoji 旅行和地点 ✈️空中运输13', 'Emoji 人类和身体 🎅虚构人物32', 'Emoji 旅行和地点 ⛲其他场所17', 'Emoji 符号 ↩️箭头21', 'Emoji 笑脸和情感 🤧病脸12', 'Emoji 食物和饮料 🍕熟食34', 'Emoji 肤色和发型 🦱发型4', 'Emoji 符号 ⚠️警告13', 'Emoji 旗帜 🇬🇧国家或地区   旗258', 'Emoji 动物和自然 🐀哺乳动物64', 'Emoji 物品 💡光线和视频16', 'Emoji 笑脸和情感 😴睡脸5', 'Emoji 符号 ‼标点7', 'Emoji 旅行和地点 🚢水路交通9', 'Emoji 动物和自然 🐛昆虫16', 'Emoji 物品 🎵音乐9', 'Emoji 符号 ⏏️视频标志24', 'Emoji 物品 🔭科技7', 'Emoji 食物和饮料 🍴餐具7', 'Emoji 旅行和地点 ⛪宗教场所6', 'Emoji 活动 🎈事件21', 'Emoji 人类和身体 ✍️动作手3', 'Emoji 符号 🚻功能标识13', 'Emoji 笑脸和情感 🤠带帽脸3', 'Emoji 人类和身体 🙋人物手势30', 'Emoji 物品 🎹乐器9', 'Emoji 符号 ♈星座13', 'Emoji 物品 👖服饰45', 'Emoji 物品 ✉️邮件13', 'Emoji 食物和饮料 🥬  蔬菜15', 'Emoji 人类和身体 🖐手掌张开9', 'Emoji 物品 ✏️书写7', 'Emoji 符号 💲货币2', 'Emoji 笑脸和情感 🤐中性脸-怀疑脸13', 'Emoji  笑脸和情 感 💩装扮脸8', 'Emoji 旅行和地点 🌋地理9', 'Emoji 物品 ✂️办公23', 'Emoji 人类和身体 👨\u200d👩\u200d👧\u200d👦家庭38', 'Emoji 笑脸和情感 😞担忧脸26', 'Emoji 人类和身体 👍合上手掌6', 'Emoji 人类和身体 🤝双手7', 'Emoji 食物和饮料 🍚亚洲食物17', 'Emoji 旅行和地点 🛎️酒店2']
canmo=[20,[],-1,512,512,"Euler a",7]


def getVar(id,data):
    with open(path+"/storage.json") as storage:
        storage=json.load(storage)
    return storage.get(id) or data

def putVar(id,data):
    with open(path+"/storage.json") as storage:
        storage=json.load(storage)
    storage.update({id:data})
    with open(path+"/storage.json","w",encoding="utf-8") as file:
        file.write(json.dumps(storage))

def getItem(id,data):
    with open(path+"/item.json") as ite:
        ite=json.load(ite)
    return ite.get(id) or data

def setItem(id,data):
    with open(path+"/item.json") as ite:
        ite=json.load(ite)
    ite.update({id:data})
    with open(path+"/item.json","w",encoding="utf-8") as file:
        file.write(json.dumps(ite))

def seach(inp, num,sf="search",yin="yinyin",yinli=[],start=0):
    sss=search if sf=="search" else find
    ddd=tags if sf=="search" else ddict
    yin=getItem(yin,{})
    yin=[yin.get(it,[]) for it in yinli]
    yin=[it for its in yin for it in its]
    yin=list(set(yin))
    if len(inp) != 0:
        input = [sss.get(item) or [] for item in list(inp)]
        index = 0
        for item in input:
            if index == 0:
                index = 1
                ss = set(item)
            else:
                ss = ss & set(item)
        ss=[it for it in list(ss) if it not in yin]
        input = [ddd[item] for item in ss]
        inputfront=[it for it in input if inp in it.get("tags") or inp in it.get("chin").replace("\\n"," ")]
        inputbehind=[it for it in input if inp not in it.get("tags") and inp not in it.get("chin").replace("\\n"," ")]
        inputfront = sorted(inputfront, key=lambda item: int(
            (item.get("num") or str(item.get("index")))), reverse=(True if sf=="search" else False))
        inputbehind = sorted(inputbehind, key=lambda item: int(
            (item.get("num") or str(item.get("index")))), reverse=(True if sf=="search" else False))
        input=inputfront+inputbehind
    else:
        ddd=[it for it in ddd if it.get("index") not in yin]
        input = sorted(ddd, key=lambda item: int(
            (item.get("num") or str(item.get("index")))), reverse=(True if sf=="search" else False))
    return [[i.get("tags")+"【"+i.get("chin").replace("\\n"," ")+"】【"+(i.get("num") or str(i.get("index")))+"】",i.get("index"),len(input)] for i in input[start:num]]

def seach1(inp,aa=0,bb=len(ddict)-1):
    if inp=="":
        return 0
    elif aa==bb:
        return aa
    elif bb-aa==1:
        if ddict[aa].get("tags")==inp:
            return aa
        elif ddict[bb].get("tags")==inp:
            return bb
        else:
            return float(aa)+0.5
    elif ddict[int((aa+bb)/2)].get("tags")==inp:
        return int((aa+bb)/2)
    elif sorted([ddict[int((aa+bb)/2)].get("tags"),inp])[0]==inp:
        return seach1(inp,aa=aa,bb=int((aa+bb)/2))
    elif sorted([ddict[int((aa+bb)/2)].get("tags"),inp])[0]==ddict[int((aa+bb)/2)].get("tags"):
        return seach1(inp,aa=int((aa+bb)/2),bb=bb)

def sseach(inp,ind):
    allthing=seach(inp=inp,num=maxmax,sf="search",yin="yinyin",yinli=[])
    for it in allthing:
        if it[0].split("【")[0]==inp:
            return ["search",it[1],ind]
    if not perfect:
        allthing=seach1(inp=inp)
        if allthing%1==0:
            return ["find",allthing,ind]

def bian(te,rep):
    if len(te)==0:
        return te
    while te[0] in rep:
        te=te[1:]
        if len(te)==0:
            break
    if len(te)==0:
        return te
    while te[len(te)-1] in rep:
        te=te[:-1]
        if len(te)==0:
            break
    return te

def text_to_check(text, num,r1818,check,yinli):
    setItem("set_num",num)
    input =[it[0] for it in seach(text, num,yinli=yinli)]
    if "R18" not in r1818:
        jian=[it[0] for it in seach("R18",maxmax)]
        input=[it for it in input if it not in jian]
    return gr.update(choices=input),[it for it in check if it in input]

def ttext_to_check(text,num,check,yin,eng):
    setItem("set_nnum",num)
    input=[it[0] for it in seach(text,num,sf="find",yin="yyinyin",yinli=yin)]
    input=[it for it in input if eng in it]
    '''start=seach1(eng)
    end=seach1(eng+"z"*10)
    start=start if start%1==0 else int(start-0.5)
    end=end if end%1==0 else int(end+0.5)
    input=list(set([i.get("tags")+"【"+i.get("chin").replace("\\n"," ")+"】【"+(i.get("num") or str(i.get("index")))+"】" for i in ddict[start:end+1]]+input))'''
    return gr.update(choices=input),[it for it in check if it in input]

def radio_to_out(li,rad,sab,cho="one_tags"):
    setItem("set_dro",li)
    yuan=getVar(cho,[])
    text = ""
    ind=0
    add=""
    hou=""
    for it in yuan:
        one = it.index("—")
        two = it.index("—", one+1)
        num = float(it[one+1:two])
        word = it.split("【")[0]
        if word[0]=="$":
            word=word.split("$")[0]
        if num < 0:
            fu = "[]"
            num = -num
        elif num > 0:
            fu = li[0:2] if li[0:3] in "()( {}(" else ("()" if word in rad else "{}")
            num -= 1
        else:
            continue
        if "【特殊符号：" in it:
            word=word.split("$")[0]
        if ind!=len(yuan)-1 and "【特殊符号：" in yuan[ind+1] and (yuan[ind+1].split("$")[0]=="|" or yuan[ind+1].split("$")[0]==":" or yuan[ind+1].split("$")[0]=="::") and "【特殊符号：" not in it and (ind==0 or set(list(yuan[ind+1].split("$")[0]))!=set(list(yuan[ind-1].split("$")[0]))):
            qian="["
        else:
            qian=""
        if "【特殊符号：" in it:
            if ind==0 or ind==len(yuan)-1 or set(list(yuan[ind+1].split("【")[0].split("$")[0]))==set(list(word)):
                add=word+" "
                hou=""
            elif word=="|" or word==":" or word=="::":
                if ind==1 or set(list(yuan[ind-2].split("【")[0].split("$")[0]))!=set(list(word)):
                    hou=""
                if ind==len(yuan)-2 or set(list(yuan[ind+2].split("【")[0].split("$")[0]))!=set(list(word)):
                    hou="]"
                add=word+" "
            else:
                add=word+" "
                hou=""
        elif num%1==0:
            num=int(num)
            add=qian+fu[0]*num+word+fu[1]*num+hou+(", " if word in sab and(ind!=len(yuan)-1 or "最后一个去逗号" not in getItem("set_little",[]))and(ind==len(yuan)-1 or "【特殊符号：" not in yuan[ind+1]) else " ")
            hou=""
        else:
            add=qian+fu[0]+word+":"+str(num+1)+fu[1]+hou+(", " if word in sab and(ind!=len(yuan)-1 or "最后一个去逗号" not in getItem("set_little",[]))and(ind==len(yuan)-1 or "【特殊符号：" not in yuan[ind+1]) else " ")
            hou=""
        text+=add
        ind+=1

    yuans=[it.split("【")[0] for it in yuan if "【特殊符号：" not in it]
    if li[0:3]=="()(":
        biansm=yuans
    elif li[0:3]=="{}(":
        biansm=[]
    elif li[0:3]=="({}":
        biansm=[it for it in rad if it in yuans]
    return text,biansm,[it for it in sab if it in yuans]

def check_to_sub(check, radio, li,su,bas,sab,little,clear=False):
    if type(check)==str:
        check=[check]
    elif check=="":
        check=[""]
    check=[addtime(it.replace("$【","【")) if "$【" in it else it for it in check]
    ssab=[it.split("【")[0].replace("?(&&","").replace("?[&&","").replace("?{&&","") for it in check if it[0]=="?"]
    check=[it[1:] if (it!="" and it[0]=="?") else it for it in check]
    check=[it if "【" in it else it+"【】【】—1—" for it in check]
    lis=["{}(使用大括号作为增强符号)", "()(使用小括号作为增强符号)","({})(使用混合括号)"]
    yuan=getVar("one_tags",[])
    cun=[it.split("【")[0] for it in yuan]
    checkan=[item.split("—")[0] for item in check]
    checkan=[item[3:] if (item[0] in "({" and item[1:3]=="&&") else item for item in checkan]
    yuan=[item for item in yuan if item.split("—")[0] not in checkan or "【特殊符号：" in item]
    check=[item if "—" in item else item+"—1—" for item in check]
    bians=list(set([item[0] for item in check if item[1:3]=="&&"]))
    if "末尾" in su:
        yuan = yuan+check
    else:
        yuan=[check+[it] if it.split("【")[0]==(radio or "").split("【")[0] else [it] for it in yuan] or [check]
        yuan=[it for its in yuan for it in its]
    fuhao={}
    [fuhao.update({it[3:].split("【")[0]:(it[0] if it[1:3]=="&&" else "&")}) for it in yuan]
    yuan=[it[3:] if (it[0] in "({" and it[1:3]=="&&") else it for it in yuan]
    yuans=[it.split("【")[0] for it in yuan if "【特殊符号：" not in it]
    if len(bians)==0:
        if li[0:3]=="()(":
            biansm=yuans
        elif li[0:3]=="{}(":
            biansm=[]
        elif li[0:3]=="({}":
            biansm=[it for it in bas if it in yuans]
    elif len(bians)==1 and bians[0]=="(" and (li[0:3]=="()(" or len(getVar("one_tags",[]))==0):
        li=lis[1]
        biansm=yuans
    elif len(bians)==1 and bians[0]=="{" and (li[0:3]=="{}(" or len(getVar("one_tags",[]))==0):
        li=lis[0]
        biansm=[]
    else:
        li=lis[2]
        biansm=[it for it in yuans if (it in bas or fuhao.get(it)=="(")]
    putVar("one_tags",yuan)
    sab=[it for it in yuans if (it not in cun or it in sab) and it not in ssab]
    bas1,bas,sba1,sab,li1,li,text,ccansteps,ccanface,ccanseed,ccanwidth,ccanheight,ccansamples,ccancfg,ccansteps,ccanface,ccanseed,ccanwidth,ccanheight,ccansamples,ccancfg=lit_to_bassab(little,biansm,sab,li,*canmo)
    setItem("set_dro",li)
    if not clear:
        return gr.update(choices=yuan),radio if radio in yuan or len(yuan)==0 else yuan[0], text,gr.update(choices=yuans),bas,li,gr.update(choices=yuans),sab
    else:
        return gr.update(choices=yuan),radio if radio in yuan or len(yuan)==0 else yuan[0], text,gr.update(choices=yuans),bas,li,gr.update(choices=yuans),sab,[]

def but_to_radio(radio, cho):
    try:
        yuan=getVar("one_tags",[])
        one = radio.index("—")
        two = radio.index("—", one+1)
        num = float(radio[one+1:two])
        if cho=="big":
            num+=1
        elif cho=="small":
            num-=1
        else:
            num=cho
        index = 0
        for it in yuan:
            if it == radio:
                radio = radio[0:one+1]+str(num)+radio[two:]
                yuan[index] = radio
                putVar("one_tags",yuan)
                return gr.update(choices=yuan, value=radio)
            index += 1
    except:
        return

def zhou_to_check(nname):
    yuan=getItem("zhoucun",{})
    if yuan.get(nname):
        yuan=yuan.get(nname)
        check=[it if type(it)==str else((tags[it[1]],it[2]) if it[0]=="search" else (ddict[it[1]],it[2])) for it in yuan[2]]
        bas=[it if type(it)==str else(tags[it[1]]["tags"] if it[0]=="search" else ddict[it[1]]["tags"]) for it in yuan[1]]
        sab=[it if type(it)==str else(tags[it[1]]["tags"] if it[0]=="search" else ddict[it[1]]["tags"]) for it in yuan[3]]
        check=[it.split("&&")[1] if type(it)==str else it[0].get("tags")+"【"+it[0].get("chin").replace("\\n"," ")+"】【"+(it[0].get("num") or str(it[0].get("index")))+"】—"+str(it[1])+"—" for it in check]
        check=[("?" if it.split("【")[0] not in sab else "")+("(" if it.split("【")[0] in bas else "{")+"&&"+it for it in check]
        return gr.update(choices=check),check,zhou_to_out(nname,check)
    else:
        return gr.update(choices=[]),[],""

def zhou_to_out(nname,check):
    putVar("zancun",[it.split("&&")[1] for it in check])
    yuan=getItem("zhoucun",{}).get(nname)
    out,bas,sab=radio_to_out(cho="zancun",li=yuan[0],rad=[it if type(it)==str else(tags[it[1]].get("tags") if it[0]=="search" else ddict[it[1]].get("tags")) for it in yuan[1]],sab=[it if type(it)==str else(tags[it[1]].get("tags") if it[0]=="search" else ddict[it[1]]).get("tags") for it in yuan[1]] if len(yuan)==2 else [it if type(it)==str else(tags[it[1]].get('tags') if it[0]=="search" else ddict[it[1]].get("tags")) for it in yuan[3]])
    return out

def out_to_cli(outp):
    try:
        pf = pd.DataFrame([outp])
        pf.to_clipboard(index=False,header=False)
    except:
        print("您可能是在前后端分离的环境下进行操作，请手动复制，感谢您的喜欢")

def delete_to_out(dele):
    putVar("one_tags",[])
    return gr.update(choices=[]),None,"",gr.update(choices=[]),[],gr.update(choices=[]),[]

def delete_one_to_out(radio,li,bas,sab):
    yuan=getVar("one_tags",[])
    yuan=list(filter(lambda a: a != radio, yuan))
    putVar("one_tags",yuan)
    yuans=[it.split("【")[0] for it in yuan if "【特殊符号" not in it]
    text,bas,sab=radio_to_out(li,bas,sab)
    return gr.update(choices=yuan),yuan[0] if len(yuan)!=0 else None,text,gr.update(choices=yuans),bas,gr.update(choices=yuans),sab

def rr(tex,nu,r1818,check,yin):
    check0,check1=text_to_check(tex,nu,r1818,check,yin)
    if "按下enter键时搜索才生效" in r1818:
        setItem("set_r1818",["按下enter键时搜索才生效"])
    else:
        setItem("set_r1818",[])
    if "emoji细分类" in r1818:
        lili=choli+emoji
    else:
        lili=choli
    if "R18" in r1818:
        return gr.update(choices=lili),check0,check1
    else:
        return gr.update(choices=[it for it in lili if "R18" not in it]),check0,check1

def cheese_to_all(warn,cheese):
    warn=warn.split("\n")[0].split("典")[1].split("的")[0]
    file=magic.get(warn).get(cheese)
    add=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin").replace("\\n"," ")+"】【"+tags[it[0]].get("num")+"】【"+it[3]+"】—"+str(it[1])+"—" for it in file.get("add")]
    reduce=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin").replace("\\n"," ")+"】【"+tags[it[0]].get("num")+"】【"+it[3]+"】—"+str(it[1])+"—" for it in file.get("reduce")]
    img=[path+"/images/magic"+"@".join(re.findall(r"\d",warn))+"/"+it for it in os.listdir(path+"/images/magic"+"@".join(re.findall(r"\d",warn))+"/") if it.split("@")[0]==cheese]
    warn="元素法典"+warn+"的各种使用技巧和提示：\n\n"+file.get("name")+"：\n\n细节："+file.get("detail")+"\n\n可改进："+file.get("progress")+"\n\n其他设置："+file.get("settings")
    imgcho=["第"+str(it+1)+"张魔法成品" for it in range(len(img))]
    addcheck=list(set([it[3] for it in file.get("add")]))
    reducecheck=list(set([it[3] for it in file.get("reduce")]))
    return gr.update(choices=add),gr.update(choices=reduce),gr.update(choices=imgcho),imgcho[0],img[0],gr.update(label=cheese),warn,add,reduce,gr.update(choices=addcheck),addcheck,gr.update(choices=reducecheck),reducecheck

def lit_to_bassab(little,bas,sab,li,ccansteps,ccanface,ccanseed,ccanwidth,ccanheight,ccansamples,ccancfg):
    setItem("set_little",little)
    yuan=[it.split("【")[0] for it in getVar("one_tags",[]) if "【特殊符号：" not in it]
    lis=["{}(使用大括号作为增强符号)", "()(使用小括号作为增强符号)","({})(使用混合括号)"]
    bigvisit="全部使用大括号(优先级低)" not in little
    smavisit="全部使用小括号(优先级高)" not in little
    if not bigvisit:
        bas=[]
        li=lis[0]
    if not smavisit:
        bas=yuan
        li=lis[1]
    sabvisit="全部后加逗号" not in little
    if not sabvisit:
        sab=yuan
    out,bas,sab=radio_to_out(li,bas,sab)
    return gr.update(visible=bigvisit and smavisit),bas,gr.update(visible=sabvisit),sab,gr.update(visible=bigvisit and smavisit),li,out,*(canmo if "其他参数为默认值" in little else [ccansteps,ccanface,ccanseed,ccanwidth,ccanheight,ccansamples,ccancfg]),gr.update(visible=("其他参数为默认值" not in little)),gr.update(visible=("其他参数为默认值" not in little)),gr.update(visible=("其他参数为默认值" not in little)),gr.update(visible=("其他参数为默认值" not in little)),gr.update(visible=("其他参数为默认值" not in little)),gr.update(visible=("其他参数为默认值" not in little)),gr.update(visible=("其他参数为默认值" not in little))

def image_appear(warning,name,num):
    num=int(num.split("第")[1].split("张")[0])-1
    warning=warning.split("\n")[0].split("典")[1].split("的")[0]
    img=[path+"/images/magic"+"@".join(re.findall(r"\d",warning))+"/"+it for it in os.listdir(path+"/images/magic"+"@".join(re.findall(r"\d",warning))+"/") if it.split("@")[0]==name]
    return img[num]

def zhou_to_cun(name,li,rad,sab):
    yuan=getVar("one_tags",[])
    yuan=[re.sub(r"\$\d+","$",it) for it in yuan]
    yuan=[("?" if it.split("【")[0] not in sab else "")+("(" if it.split("【")[0] in rad else "{")+"&&"+it for it in yuan]
    zhou=getItem("zhoucun",{})
    if name=="":
        name="您的第"+str(len(zhou.keys())+1)+"卷魔咒记载"
    zhou.update({name:[li,rad,yuan,sab]})
    setItem("zhoucun",zhou)
    onebenupdate(2)
    return zhou_to_appear(),name if len(zhou)!=0 else None,""

def tag_to_cun(name,onetag):
    alltag=getItem("tagcun",{})
    if name=="":
        name="您的第"+str(len(alltag.keys())+1)+"个魔法碎片"
    onetag=bian(onetag," ,")
    alltag.update({name:onetag})
    setItem("tagcun",alltag)
    return tag_to_appear(),name if len(alltag)!=0 else None,""

def zhou_to_appear():
    return gr.update(choices=[it for it in getItem("zhoucun",{})])

def tag_to_appear():
    return gr.update(choices=[it for it in getItem("tagcun",{})])

def zhou_del(nname):
    yuan=getItem("zhoucun",{})
    if nname in yuan.keys():
        yuan.pop(nname)
    setItem("zhoucun",yuan)
    return zhou_to_appear(),gr.update(choices=[]),[],"",([it for it in yuan][0] if len([it for it in yuan])!=0 else None)

def tag_del(nname):
    yuan=getItem("tagcun",{})
    if nname in yuan.keys():
        yuan.pop(nname)
    setItem("tagcun",yuan)
    return tag_to_appear(),"",([it for it in yuan][0] if len([it for it in yuan])!=0 else None)

def sea_cun(text,name="seasea"):
    lili=getItem(name,[])+[text]
    setItem(name,lili)
    return gr.update(choices=lili)

def yin_cun(text,name="yinyin",sf="search"):
    lili=getItem(name,{})
    llist=tags if sf=="search" else ddict
    nname=[it.get("index") for it in llist if text in it.get("tags") or text in it.get("chin").replace("\\n"," ")]
    lili.update({text:nname})
    setItem(name,lili)
    return gr.update(choices=[it for it in lili])

def sea_del(delete,name="seasea"):
    lili=getItem(name,[])
    lili=[it for it in lili if it not in delete]
    setItem(name,lili)
    return gr.update(choices=lili),[]

def yin_del(delete,name="yinyin"):
    lili=getItem(name,{})
    for it in delete:
        lili.pop(it)
    setItem(name,lili)
    return gr.update(choices=[it for it in lili]),[]

def ifinnot(word,any):
    for it in any:
        if word in it or it in word:
            return True
    return False

def yin_can(yin,text,num,r1818,check):
    if "emoji细分类" in r1818:
        lili=choli+emoji
    else:
        lili=choli
    cho=[it for it in lili if not ifinnot(it,yin)]
    if "R18" not in r1818:
        cho=[it for it in cho if "R18" not in it]
    check1,check2=text_to_check(text,num,r1818,check,yinli=yin)
    return gr.update(choices=cho),check1,check2
def ch(fn,item,name,data,warn):
    if fn(item):
        item=getItem(name,data)
        print(warn)
    setItem(name,item)
    return item
if dictperfect:
    def cun_settings(sstep,sbig,ssmall,tbig,tsmall,ttbig,ttsmall):
        sstep=ch(lambda it:it<=0,sstep,"set_step",0.000001,"你输入的步长值非法")
        sbig=ch(lambda it:it%1!=0,sbig,"set_big",20,"你输入的最大值非法")
        ssmall=ch(lambda it:it%1!=0,ssmall,"set_small",-20,"你输入的最小值非法")
        tbig=ch(lambda it:it%1!=0 or it<=0,tbig,"set_text_big",500,"你输入的最大值非法")
        tsmall=ch(lambda it:it%1!=0 or it<=0,tsmall,"set_text_small",1,"你输入的最小值非法")
        ttbig=ch(lambda it:it%1!=0 or it<=0,ttbig,"set_ttext_big",500,"你输入的最大值非法")
        ttsmall=ch(lambda it:it%1!=0 or it<=0,ttsmall,"set_ttext_small",1,"你输入的最小值非法")

        return gr.update(step=sstep,minimum=ssmall,maximum=sbig),sstep,sbig,ssmall,gr.update(minimum=tsmall,maximum=tbig),tbig,tsmall,gr.update(minimum=ttsmall,maximum=ttbig),ttbig,ttsmall
else:
    def cun_settings(sstep,sbig,ssmall,tbig,tsmall):
        sstep=ch(lambda it:it<=0,sstep,"set_step",0.000001,"你输入的步长值非法")
        sbig=ch(lambda it:it%1!=0,sbig,"set_big",20,"你输入的最大值非法")
        ssmall=ch(lambda it:it%1!=0,ssmall,"set_small",-20,"你输入的最小值非法")
        tbig=ch(lambda it:it%1!=0 or it<=0,tbig,"set_text_big",500,"你输入的最大值非法")
        tsmall=ch(lambda it:it%1!=0 or it<=0,tsmall,"set_text_small",1,"你输入的最小值非法")
        
        return gr.update(step=sstep,minimum=ssmall,maximum=sbig),sstep,sbig,ssmall,gr.update(minimum=tsmall,maximum=tbig),tbig,tsmall


def out_to_out(input,bas,sab):
    yuan=getVar("one_tags",[])
    yuan=[re.sub(r"\$\d+【","$【",it) for it in yuan]
    yuan=[("?" if it.split("【")[0] not in sab else "")+("(" if it.split("【")[0] in bas else "{")+"&&"+it for it in yuan]
    putVar("two_tags",yuan)
    return input

def change_tabs(change,warn,cheese,goodcheck,badcheck):
    good1,bad1,imagecho1,imagecho2,image1,image2,warn,good2,bad2,goodcheck1,goodcheck2,badcheck1,badcheck2=cheese_to_all(warn,cheese)
    for it in change:
        if it[3]=="改":
            yester=it[4:].split("为")[0]
            now=it[4:].split("为")[1]
            tagskind=True if it[1]=="正" else False
        elif it[3:5]=="消掉":
            xiao=it[5:]
            tagskind=True if it[1]=="正" else False
        elif it[3]=="加":
            addadd=it[4:].split("在")[0]
            qian=it[4:].split("在")

def pt_refresh(check):
    embedpath="embeddings"
    if not os.path.exists(embedpath):
        embedpath="\\".join(__file__.split("\\")[:-4])+"\\embeddings"
    if not os.path.exists(embedpath):
        lili=[]
    else:
        lili=[it.split(".")[0] for it in os.listdir(embedpath) if ".pt" in it or ".ckpt" in it]
    return gr.update(choices=lili),[it for it in check if it in lili]

def addtime(text):
    putVar("time",getVar("time",0)+1)
    return text.split("【")[0]+"$"+str(getVar("time",0))+"【"+"【".join(text.split("【")[1:])

if not tiao:
    def send_prompt(tabname,textpower,buttons,addaor=[],addtextpower=[]):
        aor=addaor+['Prompt', 'Negative prompt', 'Steps', 'Face restoration']+(["Seed"] if shared.opts.send_seed else [])+['Size-1', 'Size-2']
        modules.generation_parameters_copypaste.bind_buttons({tabname:buttons},None,"maple_"+tabname)
        textpower=textpower if shared.opts.send_seed else textpower[:4]+textpower[5:]
        textpower=addtextpower+textpower
        send_text=[(textpower[i],aor[i]) for i in range(len(aor))]
        modules.generation_parameters_copypaste.add_paste_fields("maple_"+tabname, None, send_text)

def on_ui_tabs():
    with gr.Blocks() as block:
        with gr.Column():
            li = ["{}(使用大括号作为增强符号)", "()(使用小括号作为增强符号)","({})(使用混合括号)"]
            bas1,bas2,sab1,sab2,dro1,dro2,txt,ccansteps,ccanface,ccanseed,ccanwidth,ccanheight,ccansamples,ccancfg,ccansteps,ccanface,ccanseed,ccanwidth,ccanheight,ccansamples,ccancfg=lit_to_bassab(getItem("set_little",[]),[],[],getItem("set_dro",li[0]),*canmo)
            with gr.Row():
                with gr.Column(scale=9):
                    bas=gr.CheckboxGroup(type="value",label="此处是调节大小括号混合(选中为小括号)",visible=bas1.get("visible"),elem_id="warning")
                    sab=gr.CheckboxGroup(type="value",label="此处是调节tag后逗号是否出现(可用于组合tag)",visible=sab1.get("visible"))
                    radio = gr.Radio(label="此处是已经加入的tag")
                with gr.Column(scale=1):
                    maohao=gr.Slider(minimum=getItem("set_small",-20),maximum=getItem("set_big",20),step=getItem("set_step",0.000001),label="此处可拉动选择括号权重",value=0)
                    big = gr.Button("增加选定tag权重")
                    small = gr.Button("减少选定tag权重")
                    delete=gr.Button("点我清空选中tag")
                    deleteone=gr.Button("点我删除选中tag")
                    nname=gr.Textbox(label="此处填写欲收藏组合/单个咒语名称,为空则默认格式")
                    zhoucun=gr.Button("点我保存框中文本为组合咒语")
                    tagcun=gr.Button("点我保存框中文本为单个咒语")
                    cansteps=gr.Slider(visible=("其他参数为默认值" not in getItem("set_little",[])),interactive=True,minimum=1,maximum=150,value=20,step=1,label="(调节参数)采样迭代步数")
                    canface=gr.CheckboxGroup(visible=("其他参数为默认值" not in getItem("set_little",[])),interactive=True,choices=['Restore faces'],value=[],label="(调节参数)面部修复")
                    canwidth=gr.Slider(visible=("其他参数为默认值" not in getItem("set_little",[])),interactive=True,minimum=64,maximum=2048,value=512,step=64,label="(调节参数)宽度")
                    canheight=gr.Slider(visible=("其他参数为默认值" not in getItem("set_little",[])),interactive=True,minimum=64,maximum=2048,value=512,step=64,label="(调节参数)高度")
                    canseed=gr.Number(visible=("其他参数为默认值" not in getItem("set_little",[])),value=-1,label="(调节参数)随机种子",interactive=True)
                    cancfg=gr.Slider(visible=("其他参数为默认值" not in getItem("set_little",[])),interactive=True,minimum=1,maximum=30,value=7,step=1,label="(调节参数)提示词相关性")
                    cansamples=gr.Radio(visible=("其他参数为默认值" not in getItem("set_little",[])),interactive=True,choices=["Euler a","Euler","LMS","Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM fast","DPM adaptive","LMS Karras","DPM2 Karras","DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DDIM"],value="Euler a",label="(调节参数)采样方法")
            with gr.Row():
                with gr.Column(scale=9):
                    out = gr.Textbox(lines=7, max_lines=100, label="此处是咒语输出框一(默认)(正向tags)",interactive=True)
                    negout=gr.Textbox(lines=7,max_lines=100,label="此处是咒语输出框二(反向tags)",interactive=True)
                with gr.Column(scale=1):
                    dro = gr.Dropdown(choices=li, value=dro2, interactive=True, label="此处选择增强符号形式",visible=dro1.get("visible"))
                    little=gr.CheckboxGroup(choices=["全部使用大括号(优先级低)","全部使用小括号(优先级高)","全部后加逗号","最后一个去逗号","其他参数为默认值"],value=getItem("set_little",[]),label="一些选项")
                    cli = gr.Button("点击我复制咒语文本")
                    toout=gr.Button(value="点击我将输出框一内容转移到输出框二")
                    outto=gr.Button(value="点击我将输入框二内容提交到输入框一所有tag末尾")
                    outqian=gr.Button(value="点击我将输入框二内容提交到输入框一选中tag前面")
                    txt2imgbutton=gr.Button(value="转到文生图页面并复制参数")
                    img2imgbutton=gr.Button(value="转到图生图页面并复制参数")
        with gr.Tab(label="单个咒语书柜"):
            with gr.Row():
                with gr.Column(scale=9):
                    text = gr.Textbox(lines=1, label="请在此处输入中文或英文关键词搜索单个咒语")
                    cho=gr.Radio(label="尝试一下这些大类分组吧",choices=[it for it in choli if "R18" not in it],type="value")
                with gr.Column(scale=1):
                    seabutton=gr.Button(value="保存为搜索关键词")
                    sea=gr.CheckboxGroup(choices=getItem("seasea",[]),value=[],label="此处为搜索关键词")
                    seadel=gr.Button(value="删除选中搜索关键词")
                    yinbutton=gr.Button(value="保存为隐藏关键词")
                    yin=gr.CheckboxGroup(choices=[it for it in getItem("yinyin",{})],value=[],label="此处为隐藏关键词")
                    yindel=gr.Button(value="删除选中隐藏关键词")
            with gr.Row():
                with gr.Column(scale=9):
                    check = gr.CheckboxGroup(choices=[it[0] for it in seach("", getItem("set_num",100))], label="此处是单个咒语搜索结果",value=[])
                with gr.Column(scale=1):
                    sub = gr.Button(value="在所有tag末尾提交所有单个咒语")
                    subding=gr.Button(value="在选中tag前面添加所有单个咒语")
                    r18=gr.CheckboxGroup(choices=["R18","emoji细分类","按下enter键时搜索才生效"],value=getItem("set_r1818",[]),label="一些选项")
                    num = gr.Slider(minimum=getItem("set_text_small",1), maximum=getItem("set_text_big",500), step=1,value=getItem("set_num",100), label="此处是调整搜索结果个数")
            cho.change(fn=text_to_check,inputs=[cho,num,r18,check,yin],outputs=[check,check])
            text.change(fn=lambda *it:text_to_check(*it) if "按下enter键时搜索才生效" not in it[2] else (it[3],it[3]), inputs=[text, num,r18,check,yin], outputs=[check,check])
            text.submit(fn=text_to_check, inputs=[text, num,r18,check,yin], outputs=[check,check])
            num.change(fn=text_to_check, inputs=[text, num,r18,check,yin], outputs=[check,check])
            sub.click(fn=check_to_sub, inputs=[check, radio, dro,sub,bas,sab,little], outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            subding.click(fn=check_to_sub,inputs=[check,radio,dro,subding,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            r18.change(fn=rr,inputs=[text,num,r18,check,yin],outputs=[cho,check,check])
            seabutton.click(fn=sea_cun,inputs=text,outputs=sea)
            seadel.click(fn=sea_del,inputs=sea,outputs=[sea,sea])
            sea.change(fn=lambda it:" ".join(it),inputs=sea,outputs=text)
            yinbutton.click(fn=yin_cun,inputs=text,outputs=yin)
            yindel.click(fn=yin_del,inputs=yin,outputs=[yin,yin])
            yin.change(fn=yin_can,inputs=[yin,text,num,r18,check],outputs=[cho,check,check])
        if magicperfect:
            with gr.Tab(label="元素法典卷轴"):
                mag=sorted(magic.keys())
                for item in mag:
                    with gr.Tab(label=item):
                        file=magic.get(item)
                        names=sorted([it for it in file])
                        file=file.get(names[0])
                        with gr.Row():
                            with gr.Column(scale=2):
                                warn=gr.Textbox(lines=10,value="此处是元素法典"+item+"的各种使用技巧和提示：\n\n"+names[0]+"：\n\n细节："+file.get("detail")+"\n\n可改进："+file.get("progress")+"\n\n其他设置："+file.get("settings"),interactive=False,label="此处是您的元素法典使用说明书")
                            with gr.Column(scale=1):
                                img=[path+"/images/magic"+"@".join(re.findall(r"\d",item))+"/"+it for it in os.listdir(path+"/images/magic"+"@".join(re.findall(r"\d",item))+"/") if it.split("@")[0]==names[0]]
                                imgchoi=["第"+str(it+1)+"张魔法成品" for it in range(len(img))]
                                imagecho=gr.Radio(choices=imgchoi,value=imgchoi[0],label="此处是切换示例图像")
                                image=gr.Image(value=img[0],label=names[0])
                        with gr.Row():
                            with gr.Column(scale=2):
                                cheese=gr.Radio(choices=names,label="此处是各种魔法咒语选择",value=names[0])
                            with gr.Column(scale=1):
                                goodsub=gr.Button(value="提交选中的正面咒语组合到所有tag最末尾")
                                goodsubding=gr.Button(value="提交选中的正面咒语组合到选定tag前面")
                                badsub=gr.Button(value="提交选中的负面咒语组合到所有tag最末尾")
                                badsubding=gr.Button(value="提交选中的负面咒语组合到选定tag前面")
                        with gr.Row():
                            goodche=list(set([it[3] for it in magic.get(item).get(names[0]).get("add")]))
                            badche=list(set([it[3] for it in magic.get(item).get(names[0]).get("reduce")]))
                            goodcho=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin").replace("\\n"," ")+"】【"+tags[it[0]].get("num")+"】【"+it[3]+"】—"+str(it[1])+"—" for it in magic.get(item).get(names[0]).get("add")]
                            badcho=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin").replace("\\n"," ")+"】【"+tags[it[0]].get("num")+"】【"+it[3]+"】—"+str(it[1])+"—" for it in magic.get(item).get(names[0]).get("reduce")]
                            with gr.Column():
                                goodcheck=gr.CheckboxGroup(choices=goodche,value=goodche,label="此处是该魔法正向tag分类")
                                good=gr.CheckboxGroup(choices=goodcho,label="此处是该魔法正向tag",value=goodcho)
                            with gr.Column():
                                badcheck=gr.CheckboxGroup(choices=badche,value=badche,label="此处是该魔法负向tag分类")
                                bad=gr.CheckboxGroup(choices=badcho,label="此处是该魔法负向tag",value=badcho)

                        cheese.change(fn=cheese_to_all,inputs=[warn,cheese],outputs=[good,bad,imagecho,imagecho,image,image,warn,good,bad,goodcheck,goodcheck,badcheck,badcheck])
                        imagecho.change(fn=image_appear,inputs=[warn,cheese,imagecho],outputs=image)
                        goodcheck.change(fn=lambda its,war,chee:[it for it in cheese_to_all(war,chee)[0].get("choices") if ifinnot(it,["【"+i+"】" for i in its])],inputs=[goodcheck,warn,cheese],outputs=good)
                        badcheck.change(fn=lambda its,war,chee:[it for it in cheese_to_all(war,chee)[1].get("choices") if ifinnot(it,["【"+i+"】" for i in its])],inputs=[badcheck,warn,cheese],outputs=bad)
                        goodsub.click(fn=check_to_sub,inputs=[good,radio,dro,goodsub,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
                        badsub.click(fn=check_to_sub,inputs=[bad,radio,dro,badsub,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
                        goodsubding.click(fn=check_to_sub,inputs=[good,radio,dro,goodsubding,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
                        badsubding.click(fn=check_to_sub,inputs=[bad,radio,dro,badsubding,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
        if dictperfect:
            with gr.Tab(label="遗失魔法碎片"):
                with gr.Row():
                    with gr.Column(scale=9):
                        engtext=gr.Textbox(lines=1,label="请在此处输入关键词筛选搜索结果(需按下enter键才能生效)")
                        ttext=gr.Textbox(lines=1,label="请在此处输入中文关键词搜索可能被遗失的魔法碎片")
                        ccheck=gr.CheckboxGroup(choices=[it[0] for it in seach("",getItem("set_nnum",100),sf="find",yin="yyinyin")],label="此处是搜索结果",value=[])
                    with gr.Column(scale=1):
                        ssub = gr.Button(value="在所有tag末尾提交可能遗失的魔法碎片")
                        ssubding=gr.Button(value="在选中tag前面添加可能遗失的魔法碎片")
                        nnum = gr.Slider(minimum=getItem("set_ttext_small",1), maximum=getItem("set_ttext_big",500), step=1,value=getItem("set_nnum",100), label="此处是调整搜索结果个数")
                        sseabutton=gr.Button(value="保存为搜索关键词")
                        ssea=gr.CheckboxGroup(choices=getItem("sseasea",[]),value=[],label="此处为搜索关键词")
                        sseadel=gr.Button(value="删除选中搜索关键词")
                        yyinbutton=gr.Button(value="保存为隐藏关键词")
                        yyin=gr.CheckboxGroup(choices=[it for it in getItem("yyinyin",{})],value=[],label="此处为隐藏关键词")
                        yyindel=gr.Button(value="删除选中隐藏关键词")
                engtext.submit(fn=ttext_to_check,inputs=[ttext,nnum,ccheck,yyin,engtext],outputs=[ccheck,ccheck])
                ttext.change(fn=ttext_to_check,inputs=[ttext,nnum,ccheck,yyin,engtext],outputs=[ccheck,ccheck])
                ttext.submit(fn=ttext_to_check,inputs=[ttext,nnum,ccheck,yyin,engtext],outputs=[ccheck,ccheck])
                nnum.change(fn=ttext_to_check,inputs=[ttext,nnum,ccheck,yyin,engtext],outputs=[ccheck,ccheck])
                ssub.click(fn=check_to_sub,inputs=[ccheck,radio,dro,ssub,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
                ssubding.click(fn=check_to_sub,inputs=[ccheck,radio,dro,ssubding,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
                sseabutton.click(fn=lambda text:sea_cun(text,name="sseasea"),inputs=ttext,outputs=ssea)
                sseadel.click(fn=lambda delete:sea_del(delete,name="sseasea"),inputs=ssea,outputs=[ssea,ssea])
                ssea.change(fn=lambda it:" ".join(it),inputs=ssea,outputs=ttext)
                yyinbutton.click(fn=lambda text:yin_cun(text,name="yyinyin",sf="find"),inputs=ttext,outputs=yyin)
                yyindel.click(fn=lambda delete:yin_del(delete,name="yyinyin"),inputs=yyin,outputs=[yyin,yyin])
                yyin.change(fn=lambda yin,text,num,check:ttext_to_check(text,num,check,yin),inputs=[yyin,ttext,nnum,ccheck],outputs=[ccheck,ccheck])
        with gr.Tab(label="你的私藏禁术"):
            with gr.Row():
                with gr.Tab(label="自定义收藏咒语组合"):
                    zhouyu=[it for it in getItem("zhoucun",{})]
                    with gr.Row():
                        zhoubutton=gr.Button(value="将选中咒语置于末尾")
                        zhouqian=gr.Button(value="将选中咒语置于选定tag前")
                        zhoudel=gr.Button(value="点我删除选中咒语")
                    zhouradio=gr.Radio(label="此处是你的收藏咒语组合列表",choices=zhouyu,value=zhouyu[0] if len(zhouyu)!=0 else None)
                    check1,check2,out1=zhou_to_check(zhouyu[0] if len(zhouyu)!=0 else None)
                    zhoucheck=gr.CheckboxGroup(label="此处是你选中的咒语组合列表",choices=check2,value=check2)
                    zhouout=gr.Textbox(label="此处是你选中的咒语组合",lines=10,max_lines=100,value=out1)
                with gr.Tab(label="自定义收藏单个咒语"):
                    alltag=[it for it in getItem("tagcun",{})]
                    with gr.Row():
                        tagbutton=gr.Button(value="将选中单个咒语置于末尾")
                        tagqian=gr.Button(value="将选中单个咒语置于选定tag前")
                        tagdel=gr.Button(value="点我删除选中单个咒语")
                    tagradio=gr.Radio(label="此处是你的单个咒语收藏列表",choices=alltag,value=alltag[0] if len(alltag)!=0 else None)
                    tagout=gr.Textbox(label="此处是你选中的单个咒语",lines=10,max_lines=100,value=getItem("tagcun",{}).get(alltag[0] if len(alltag)!=0 else None))
                with gr.Tab(label="自定义embeddings中咒语"):
                    with gr.Row():
                        ptbutton=gr.Button(value="将选中embedding置于末尾")
                        ptqian=gr.Button(value="将选中embedding置于选定tag前")
                        ptrefresh=gr.Button(value="重新加载embedding")
                    ptcheck=gr.CheckboxGroup(choices=pt_refresh([])[0].get("choices"),value=[],label="此处是你的embeddings列表")
                with gr.Tab(label="常见占位符与高级吟唱"):
                    fenbu="""介绍全部基于编纂本篇时推出的最新版 WEB-UI，对于 NAIFU 或较旧版 WEB-UI 可能不适用。
首先介绍分步描绘的各种形式：
[from:to:step]
[from::step] (to 为空)
[:to:step] (from 为空)
[to:step] (奇怪但没问题的格式，非常不建议)
它的作用是让 prompt 在达到 step 之前被视为 from，在达到后视为 to。若是在对应位置留空则视为无对应元素。step 为大于 1 的整数时表示步数，为小于 1 的正小数时表示总步数的百分比。
比如 a girl with [green hair:red hair flower:0.2] 会在前 20% 步数被视为 a girl with green hair，在后 80% 步数被视为 a girl with red hair flower。需要注意这两个描述之间的兼容性和覆盖——在步数合适的情况下，最后形成的人物会拥有绿色头发和红色花饰，但也可能因为颜色溢出导致头发也变为红色，毕竟后 80% 没有绿色头发的限定，AI 完全可以自己理解一个随机的发色。
在最新版中，分步描绘可以嵌套，形如 [from:[to:end:step2]:step1] 的语句是可以被正确识别的。且分步描绘现在支持逗号分割，形如 [1 girl, red hair: 2 girls, white hair:0.3] 的语句也可以被正确识别。
分步描绘不特别擅长细化细节，与其分步描绘不如将细化部分直接写入持续生效的部分。分步描绘更擅长在画面初期建立引导，大幅影响后续构图或画面生成。
需要注意的是，分步描绘具有视觉延后性——当要求 AI 在比如第 20 步开始描绘另一个不同的物体时，可能在比如第 24 步(或更晚)才能从人眼视觉上感知到另一个物体勉强出现在画面中。这是因为 AI 看待图片的方式和人眼看待图片的方式不同，在 AI 的认知里图片已经初具新物体的特性的时候，人眼可能依然看不出来。"""
                    shushu={":【特殊符号：分步描绘】":fenbu,
                    "::【特殊符号：分步描绘】":fenbu,
                    "AND【特殊符号：强调咏唱】":"""介绍全部基于编纂本篇时推出的最新版 WEB-UI，对于 NAIFU 或较旧版 WEB-UI 可能不适用。
短句咏唱(AND 强调咏唱)：
masterpiece, best quality, 1 girl, (blue eyes) AND (yellow hair), (white clothes) AND (red skirt) AND (black leggings), sitting, full body
注意短句咏唱的 AND 必须是三个大写字母，AND 两侧的小括号是不必要的(但建议加上)，这是一个专用语法，不过因为效果仍未明晰所以不单独介绍。此外，该语法并不能应用于所有采样方法，例如 DDIM 就不支持 AND，会导致报错""",
                    "|【特殊符号：融合描绘】":"""介绍全部基于编纂本篇时推出的最新版 WEB-UI，对于 NAIFU 或较旧版 WEB-UI 可能不适用。
然后介绍融合描绘的两种形式：
[A | B]
[A:w1 | B:w2]
它们还有分别对应的可无限延长版：
[A | B | C | …]
[A:w1 | B:w2 | C:w3 | …]
对于形如 [A | B] 的第一种，AI 将在第一步画 A、第二步画 B、第三步画 A…交替进行。而对于无限延长版，则变为第一步画 A、第二步画 B、第三步画 C…循环往复交替进行。
对于形如 [A:w1 | B:w2] 的第二种带权重版本，截至这句话被写下时仍由 NAIFU 端独占(且本语法在 NAIFU 端的中括号是不必要的)，它的实际效果不是先画 w1 步 A 然后再画 w2 步 B，虽然成品效果类似。若在 WEB-UI 端上强行使用则会导致权重数字被作为文本读取，虽然会让画面变得不同但实际上并非加权导致的效果。它的运作方式和双端都支持的 [A | B] 略有不同但效果类似，相较而言有着支持自定义比例的独特优势。
当然，WEB-UI 有着看上去类似的 [(A:w1) | (B:w2)] 语法，但它的本质其实是嵌套了一层加权，也同样不是可以自定义各部分的步数。这样的加权是对于整个咒语而言而非对于中括号内的其它部分而言的，作用域不同，所以笔者不认为这和 NAIFU 端的写法完全相同。
融合描绘不可嵌套，但同样支持逗号分割。融合描绘擅长将两种事物混合为一起，比如 a [dog | frog] in black background。""",
                    "(super f【特殊符号：占位符】":"(super f*ck cool)的前半部分，请自行填补中间单词，作用是给画面增加张力",
                    "ck cool)【特殊符号：占位符】":"(super f*ck cool)的后半部分，请自行填补中间单词，作用是给画面增加张力",
                    "(tokyo takedown)【特殊符号：占位符】":"给画面增加科幻感和城市感",
                    "(ai is sb)【特殊符号：占位符】":"稍微加一点点科技感，更多起到占用符的作用，给其他tag缓冲的空间",
                    "+【特殊符号：反复渲染】":"反复渲染（多一个加号多反复一次），一般公式用法：,(+++++(tag)//),",
                    "/【特殊符号：简易隔离】":"简易隔离tag污染，一般公式用法：,(+++++(tag)//),",
                    "【无】":"占位，什么也没有"}
                    with gr.Row():
                        shubutton=gr.Button(value="将选中特殊字符置于末尾")
                        shuqian=gr.Button(value="将选中特殊字符置于选定tag前")
                    shuradio=gr.Radio(choices=[it for it in shushu],value=[it for it in shushu][0],label="此处是常见占位符与特殊字符列表")
                    shuout=gr.Textbox(label="此处是该特殊字符的说明",lines=10,max_lines=100,value=[shushu.get(it) for it in shushu][0])

            zhouradio.change(fn=zhou_to_check,inputs=zhouradio,outputs=[zhoucheck,zhoucheck,zhouout])
            zhoucheck.change(fn=zhou_to_out,inputs=[zhouradio,zhoucheck],outputs=zhouout)
            tagradio.change(fn=lambda it:getItem("tagcun",{}).get(it),inputs=tagradio,outputs=tagout)
            zhoubutton.click(fn=check_to_sub,inputs=[zhoucheck,radio,dro,zhoubutton,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            zhouqian.click(fn=check_to_sub,inputs=[zhoucheck,radio,dro,zhouqian,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            zhoudel.click(fn=zhou_del,inputs=zhouradio,outputs=[zhouradio,zhoucheck,zhoucheck,zhouout,zhouradio])
            tagbutton.click(fn=check_to_sub,inputs=[tagout,radio,dro,tagbutton,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            tagqian.click(fn=check_to_sub,inputs=[tagout,radio,dro,tagqian,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            tagdel.click(fn=tag_del,inputs=tagradio,outputs=[tagradio,tagout,tagradio])
            ptbutton.click(fn=check_to_sub,inputs=[ptcheck,radio,dro,ptbutton,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            ptqian.click(fn=check_to_sub,inputs=[ptcheck,radio,dro,ptqian,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            ptrefresh.click(fn=pt_refresh,inputs=ptcheck,outputs=[ptcheck,ptcheck])
            shubutton.click(fn=lambda *it:check_to_sub(addtime(it[0]),*it[1:]),inputs=[shuradio,radio,dro,shubutton,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            shuqian.click(fn=lambda *it:check_to_sub(addtime(it[0]),*it[1:]),inputs=[shuradio,radio,dro,shuqian,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
            shuradio.change(fn=lambda it:shushu.get(it),inputs=shuradio,outputs=shuout)

        with gr.Tab(label="基础草药材料"):
            cun=gr.Button(value="点我保存所有设置")
            with gr.Row():
                with gr.Column():
                    sstep=gr.Number(value=getItem("set_step",0.000001),label="此处设置tag调整权重的步长")
                    sbig=gr.Number(value=getItem("set_big",20),label="此处设置tag调整权重的最大值")
                    ssmall=gr.Number(value=getItem("set_small",-20),label="此处设置tag调整权重的最小值")
                with gr.Column():
                    tbig=gr.Number(value=getItem("set_text_big",500),label="此处设置单个咒语书柜搜索结果的最大值")
                    tsmall=gr.Number(value=getItem("set_text_small",1),label="此处设置单个咒语书柜搜索结果的最小值")
                if dictperfect:
                    with gr.Column():
                        ttbig=gr.Number(value=getItem("set_ttext_big",500),label="此处设置遗失魔法碎片搜索结果的最大值")
                        ttsmall=gr.Number(value=getItem("set_ttext_small",1),label="此处设置遗失魔法碎片搜索结果的最小值")
            if dictperfect:
                cun.click(fn=cun_settings,inputs=[sstep,sbig,ssmall,tbig,tsmall,ttbig,ttsmall],outputs=[maohao,sstep,sbig,ssmall,num,tbig,tsmall,nnum,ttbig,ttsmall])
            else:
                cun.click(fn=cun_settings,inputs=[sstep,sbig,ssmall,tbig,tsmall],outputs=[maohao,sstep,sbig,ssmall,num,tbig,tsmall])

        delete.click(fn=delete_to_out,inputs=delete,outputs=[radio,radio,out,bas,bas,sab,sab])
        deleteone.click(fn=delete_one_to_out,inputs=[radio,dro,bas,sab],outputs=[radio,radio,out,bas,bas,sab,sab])
        big.click(fn=lambda it:but_to_radio(radio=it,cho="big"), inputs=radio, outputs=radio)
        small.click(fn=lambda it:but_to_radio(radio=it,cho="small"), inputs=radio, outputs=radio)
        maohao.change(fn=lambda radio,maohao:but_to_radio(radio=radio,cho=maohao),inputs=[radio,maohao],outputs=radio)
        radio.change(fn=radio_to_out, inputs=[dro,bas,sab], outputs=[out,bas,sab])
        bas.change(fn=radio_to_out,inputs=[dro,bas,sab],outputs=[out,bas,sab])
        sab.change(fn=radio_to_out,inputs=[dro,bas,sab],outputs=[out,bas,sab])
        dro.change(fn=radio_to_out, inputs=[dro,bas,sab], outputs=[out,bas,sab])
        cli.click(fn=out_to_cli, inputs=out,outputs=[])
        zhoucun.click(fn=zhou_to_cun,inputs=[nname,dro,bas,sab],outputs=[zhouradio,zhouradio,nname])
        tagcun.click(fn=tag_to_cun,inputs=[nname,out],outputs=[tagradio,tagradio,nname])
        little.change(fn=lit_to_bassab,inputs=[little,bas,sab,dro,cansteps,canface,canseed,canwidth,canheight,cansamples,cancfg],outputs=[bas,bas,sab,sab,dro,dro,out,cansteps,canface,canseed,canwidth,canheight,cansamples,cancfg,cansteps,canface,canseed,canwidth,canheight,cansamples,cancfg])
        toout.click(fn=out_to_out,inputs=[out,bas,sab],outputs=negout)
        outto.click(fn=lambda *it:check_to_sub(getVar("two_tags",[]),*it),inputs=[radio,dro,outto,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
        outqian.click(fn=lambda *it:check_to_sub(getVar("two_tags",[]),*it),inputs=[radio,dro,outqian,bas,sab,little],outputs=[radio,radio,out,bas,bas,dro,sab,sab])
        if not tiao:
            try:
                send_prompt("txt2img",[out,negout,cansteps,canface,canseed,canwidth,canheight],txt2imgbutton,addaor=["Sampler","CFG scale"],addtextpower=[cansamples,cancfg])
                send_prompt("img2img",[out,negout,cansteps,canface,canseed,canwidth,canheight],img2imgbutton,addaor=["Sampler","CFG scale"],addtextpower=[cansamples,cancfg])
            except:
                print("添加联系失败，转移参数功能不起作用，请手动复制")

    return [(block,"maple的tag选择器","maple_tags")]

# 版本更新函数
nowben=3
def benupdate(fromthing,tothing):
    for it in range(fromthing,tothing):
        onebenupdate(it)
    setItem("your_ben",tothing)

def onebenupdate(benben):
    if benben==2:
        zhoucun=getItem("zhoucun",{})
        for it in zhoucun:
            item=zhoucun.get(it)
            item=[item[0]]+[[sseach(re.sub(r"\??[\(\[\{]?&{0,2}","",its.split("【")[0]),float(its.split("—")[1] if "—" in its else 0)) or its for its in itss] for itss in item[1:]]
            if len(item)==3:
                item.append([it if type(it)==list else re.sub(r"\??[\(\[\{]?&{0,2}","",it.split("【")[0]) for it in item[2]])
            zhoucun.update({it:item})
        setItem("zhoucun",zhoucun)
    elif benben>2:
        raise
benupdate(getItem("your_ben",2),nowben)

if not tiao:
    script_callbacks.on_ui_tabs(on_ui_tabs)
else:
    on_ui_tabs()[0][0].launch()