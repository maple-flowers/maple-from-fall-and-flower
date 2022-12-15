import gradio as gr
import json
import os
import pandas as pd
import re

from modules import script_callbacks,scripts

path="extensions/maple-from-fall-and-flower/scripts"
if not os.path.exists(path+"/search.json"):
    path="\\".join(__file__.split("\\")[:-1])

with open(path+"/search.json") as search:
    search = json.load(search)
with open(path+"/tags.json") as tags:
    tags = json.load(tags)
with open(path+"/storage.json", "w", encoding="utf-8") as storage:
    storage.write("{}")
with open(path+"/magic.json") as magic:
    magic=json.load(magic)
with open(path+"/dict.json") as ddict:
    ddict=json.load(ddict)
with open (path+"/find.json") as find:
    find=json.load(find)

choli=["常用 优化Tag","常用 其他Tag","常用 R18Tag","环境 朝朝暮暮","环境 日月星辰","环境 天涯海角","风格","非emoij的人物","角色","头发&发饰 长度","头发&发饰 颜色","头发&发饰 发型","头发&发型 辫子","头发&发型 刘海/其他","头发&发型 发饰","五官&表情 常用","五官&表情 R18","眼睛 颜色","眼睛 状态","眼睛 其他","身体 胸","身体 R18","服装 衣服","服装 R18","袜子&腿饰 袜子","袜子&腿饰 长筒袜","袜子&腿饰 连裤袜","袜子&腿饰 腿饰&组合","袜子&腿饰 裤袜","袜子&腿饰 R18","鞋 鞋子","装饰 装饰","动作 动作","动作 头发相关","动作 R18","Emoij😊 表情","Emoij😊 人物","Emoij😊 手势","Emoij😊 日常","Emoij😊 动物","Emoij😊 植物","Emoij😊 自然","Emoij😊 食物","R18 ","人体","姿势","发型","表情","眼睛","衣服","饰品","袜子","风格(画质)","环境","背景","物品"]

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

def seach(input, num,sf="search"):
    sss=search if sf=="search" else find
    ddd=tags if sf=="search" else ddict
    if len(input) != 0:
        input = [sss.get(item) or [] for item in list(input)]
        index = 0
        for item in input:
            if index == 0:
                index = 1
                ss = set(item)
            else:
                ss = ss & set(item)
        input = [ddd[item] for item in list(ss)]
        input = sorted(input, key=lambda item: int(
            (item.get("num") or str(item.get("index")))), reverse=(True if sf=="search" else False))
    else:
        input = sorted(ddd, key=lambda item: int(
            (item.get("num") or str(item.get("index")))), reverse=(True if sf=="search" else False))
    return [i.get("tags")+"【"+i.get("chin")+"】【"+(i.get("num") or str(i.get("index")))+"】" for i in input[0:num]]

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

def text_to_check(text, num,r1818,check):
    input = seach(text, num)
    if "R18" not in r1818:
        jian=seach("R18",40000)
        input=[it for it in input if it not in jian]
    return gr.update(choices=input),[it for it in check if it in input]

def ttext_to_check(text,num,check):
    input=seach(text,num,sf="find")
    return gr.update(choices=input),[it for it in check if it in input]

def radio_to_out(li,rad,cho="one_tags"):
    yuan=getVar(cho,[])
    text = ""
    for it in yuan:
        one = it.index("—")
        two = it.index("—", one+1)
        num = float(it[one+1:two])
        word = it.split("【")[0]
        if num < 0:
            fu = "[]"
            num = -num
        elif num > 0:
            fu = li[0:2] if li[0:3] in "()( {}(" else ("()" if word in rad else "{}")
            num -= 1
        else:
            continue
        
        if num%1==0:
            num=int(num)
            text += fu[0]*num+word+fu[1]*num+", "
        else:
            text+=fu[0]+word+":"+str(num+1)+fu[1]+", "
    yuans=[it.split("【")[0] for it in yuan]
    if li[0:3]=="()(":
        biansm=yuans
    elif li[0:3]=="{}(":
        biansm=[]
    elif li[0:3]=="({}":
        biansm=[it for it in rad if it in yuans]
    return text,biansm

def check_to_sub(check, radio, li,su,bas):
    if type(check)==str:
        check=[check+"【】【】—1—"]
    lis=["{}(使用大括号作为增强符号)", "()(使用小括号作为增强符号)","({})(使用混合括号)"]
    yuan=getVar("one_tags",[])
    checkan=[item.split("—")[0] for item in check]
    checkan=[item[3:] if (item[0] in "({" and item[1:3]=="&&") else item for item in checkan]
    yuan=[item for item in yuan if item.split("—")[0] not in checkan]
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
    putVar("one_tags",yuan)

    yuans=[it.split("【")[0] for it in yuan]
    
    if len(bians)==0:
        if li[0:3]=="()(":
            biansm=yuans
        elif li[0:3]=="{}(":
            biansm=[]
        elif li[0:3]=="({}":
            biansm=[it for it in bas if it in yuans]
    elif len(bians)==1 and bians[0]=="(" and li[0:3]=="()(":
        biansm=yuans
    elif len(bians)==1 and bians[0]=="{" and li[0:3]=="{}(":
        biansm=[]
    else:
        li=lis[2]
        biansm=[it for it in yuans if (it in bas or fuhao.get(it)=="(")]
    text,bia=radio_to_out(li,bas)
    if radio:
        return gr.update(choices=yuan), text,gr.update(choices=yuans),biansm,li
    elif not check:
        return gr.update(choices=yuan), text,gr.update(choices=yuans),biansm,li
    else:
        return gr.update(choices=yuan, value=yuan[0]), text,gr.update(choices=yuans),biansm,li

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
        check=yuan[2]
        return gr.update(choices=check),check,zhou_to_out(nname,check)
    else:
        return gr.update(choices=[]),[],""

def zhou_to_out(nname,check):
    putVar("zancun",check)
    yuan=getItem("zhoucun",{}).get(nname)
    out,bas=radio_to_out(cho="zancun",li=yuan[0],rad=yuan[1])
    return out

def big_to_radio(radio):
    return but_to_radio(radio=radio, cho="big")

def small_to_radio(radio):
    return but_to_radio(radio=radio, cho="small")

def mao_to_radio(radio,maohao):
    return but_to_radio(radio=radio,cho=maohao)

def out_to_cli(outp):
    try:
        pf = pd.DataFrame([outp])
        pf.to_clipboard(index=False,header=False)
        return outp
    except:
        return outp

def delete_to_out(dele):
    putVar("one_tags",[])
    return gr.update(choices=[]),[],"",gr.update(choices=[]),[]

def delete_one_to_out(radio, dro, bas):
    tags = getVar("one_tags", [])
    new_tags = list(filter(lambda a: a != radio, tags))
    putVar("one_tags",new_tags)
    new_out, _ = radio_to_out(dro, bas)
    return gr.update(choices=new_tags, value=None), new_out, gr.update(choices=list(map(lambda a: re.sub('【.*$', '', a), new_tags)))

def rr(tex,nu,r1818,check):
    check0,check1=text_to_check(tex,nu,r1818,check)
    if "R18" in r1818:
        return gr.update(choices=choli),check0,check1
    else:
        return gr.update(choices=[it for it in choli if "R18" not in it]),check0,check1

def cheese_to_all(warn,cheese):
    warn=warn.split("\n")[0].split("典")[1].split("的")[0]
    file=magic.get(warn).get(cheese)
    add=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin")+"】—"+str(it[1])+"—" for it in file.get("add")]
    reduce=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin")+"】—"+str(it[1])+"—" for it in file.get("reduce")]
    img=[path+"/images/magic"+"@".join(re.findall(r"\d",warn))+"/"+it for it in os.listdir(path+"/images/magic"+"@".join(re.findall(r"\d",warn))+"/") if it.split("@")[0]==cheese]
    warn="元素法典"+warn+"的各种使用技巧和提示：\n"+file.get("name")+"：\n细节："+file.get("detail")+"\n可改进："+file.get("progress")+"\n其他设置："+file.get("settings")
    imgcho=["第"+str(it+1)+"张魔法成品" for it in range(len(img))]
    return gr.update(choices=add),gr.update(choices=reduce),gr.update(choices=imgcho),imgcho[0],img[0],gr.update(label=cheese),warn,add,reduce

def image_appear(warning,name,num):
    num=int(num.split("第")[1].split("张")[0])-1
    warning=warning.split("\n")[0].split("典")[1].split("的")[0]
    img=[path+"/images/magic"+"@".join(re.findall(r"\d",warning))+"/"+it for it in os.listdir(path+"/images/magic"+"@".join(re.findall(r"\d",warning))+"/") if it.split("@")[0]==name]
    return img[num]

def zhou_to_cun(name,li,rad):
    yuan=getVar("one_tags",[])
    zhou=getItem("zhoucun",{})
    if name=="":
        name="您的第"+str(len(zhou.keys())+1)+"卷魔咒记载"
    zhou.update({name:[li,rad,yuan]})
    setItem("zhoucun",zhou)
    return zhou_to_appear(),name if len(zhou)!=0 else None

def tag_to_cun(name,onetag):
    alltag=getItem("tagcun",{})
    if name=="":
        name="您的第"+str(len(alltag.keys())+1)+"个魔法碎片"
    onetag=bian(onetag," ,")
    alltag.update({name:onetag})
    setItem("tagcun",alltag)
    return tag_to_appear(),name if len(alltag)!=0 else None

def zhou_to_appear():
    return gr.update(choices=[it for it in getItem("zhoucun",{})])

def tag_to_appear():
    return gr.update(choices=[it for it in getItem("tagcun",{})])

def zhou_del(nname):
    yuan=getItem("zhoucun",{})
    if nname in yuan.keys():
        yuan.pop(nname)
    setItem("zhoucun",yuan)
    return zhou_to_appear(),gr.update(choices=[]),[],"",(yuan[0] if len(yuan)!=0 else None)

def tag_del(nname):
    yuan=getItem("tagcun",{})
    if nname in yuan.keys():
        yuan.pop(nname)
    setItem("tagcun",yuan)
    return tag_to_appear(),"",(yuan[0] if len(yuan)!=0 else None)

def on_ui_tabs():
    with gr.Blocks() as block:
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=9):
                    bas=gr.CheckboxGroup(type="value",label="此处是调节大小括号混合(选中为小括号)",visible=True)
                    # sab=gr.CheckboxGroup(type="value",label="此处是调节tag后逗号是否出现(可用于组合tag)",visible=True)
                    radio = gr.Radio(type="value", label="此处是已经加入的tag")
                with gr.Column(scale=1):
                    maohao=gr.Slider(minimum=-20,maximum=20,step=0.001,label="此处可拉动选择括号权重",value=0)
                    big = gr.Button("增加选定tag权重")
                    small = gr.Button("减少选定tag权重")
                    deleteOne = gr.Button("点我删除选中tag")
                    delete=gr.Button("点我清空选中tag")
                    nname=gr.Textbox(label="此处填写欲收藏组合/单个咒语名称,为空则默认格式")
                    zhoucun=gr.Button("点我保存框中文本为组合咒语")
                    tagcun=gr.Button("点我保存框中文本为单个咒语")
            with gr.Row():
                with gr.Column(scale=9):
                    out = gr.Textbox(lines=7, max_lines=100, label="此处是输出的咒语",interactive=True)
                with gr.Column(scale=1):
                    li = ["{}(使用大括号作为增强符号)", "()(使用小括号作为增强符号)","({})(使用混合括号)"]
                    dro = gr.Dropdown(choices=li, value=li[0], interactive=True, label="此处选择增强符号形式")
                    cli = gr.Button("点击我复制咒语文本")
        
        with gr.Tab(label="单个咒语书柜"):
            text = gr.Textbox(lines=1, label="请在此处输入中文或英文关键词搜索单个咒语")
            cho=gr.Radio(label="尝试一下这些大类分组吧",choices=[it for it in choli if "R18" not in it],type="value")
            with gr.Row():
                with gr.Column(scale=9):
                    check = gr.CheckboxGroup(choices=seach("", 100), label="此处是单个咒语搜索结果",value=[])
                with gr.Column(scale=1):
                    sub = gr.Button(value="在所有tag末尾提交所有单个咒语")
                    subding=gr.Button(value="在选中tag前面添加所有单个咒语")
                    r18=gr.CheckboxGroup(choices=["R18"],value=[],label="一些选项")
                    num = gr.Slider(minimum=1, maximum=500, step=1,value=100, label="此处是调整搜索结果个数")
            cho.change(fn=lambda it:it,inputs=cho,outputs=text)
            text.change(fn=text_to_check, inputs=[text, num,r18,check], outputs=[check,check])
            num.change(fn=text_to_check, inputs=[text, num,r18,check], outputs=[check,check])
            sub.click(fn=check_to_sub, inputs=[check, radio, dro,sub,bas], outputs=[radio, out,bas,bas,dro])
            subding.click(fn=check_to_sub,inputs=[check,radio,dro,subding,bas],outputs=[radio,out,bas,bas,dro])
            r18.change(fn=rr,inputs=[text,num,r18,check],outputs=[cho,check,check])
        with gr.Tab(label="元素法典卷轴"):
            mag=sorted(magic.keys())
            for item in mag:
                with gr.Tab(label=item):
                    file=magic.get(item)
                    names=sorted([it for it in file])
                    file=file.get(names[0])
                    with gr.Row():
                        with gr.Column(scale=2):
                            warn=gr.Textbox(lines=10,value="此处是元素法典"+item+"的各种使用技巧和提示：\n"+names[0]+"：\n细节："+file.get("detail")+"\n可改进："+file.get("progress")+"\n其他设置："+file.get("settings"),interactive=False,label="此处是您的元素法典/咒语组合书柜目录")
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
                        goodcho=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin")+"】【"+tags[it[0]].get("num")+"】—"+str(it[1])+"—" for it in magic.get(item).get(names[0]).get("add")]
                        badcho=[(it[2] if it[2]=="" else it[2]+"&&")+tags[it[0]].get("tags")+"【"+tags[it[0]].get("chin")+"】【"+tags[it[0]].get("num")+"】—"+str(it[1])+"—" for it in magic.get(item).get(names[0]).get("reduce")]
                        good=gr.CheckboxGroup(choices=goodcho,label="此处是该魔法正向tag",value=goodcho)
                        bad=gr.CheckboxGroup(choices=badcho,label="此处是该魔法负向tag",value=badcho)

                    cheese.change(fn=cheese_to_all,inputs=[warn,cheese],outputs=[good,bad,imagecho,imagecho,image,image,warn,good,bad])
                    imagecho.change(fn=image_appear,inputs=[warn,cheese,imagecho],outputs=image)
                    goodsub.click(fn=check_to_sub,inputs=[good,radio,dro,goodsub,bas],outputs=[radio,out,bas,bas,dro])
                    badsub.click(fn=check_to_sub,inputs=[bad,radio,dro,badsub,bas],outputs=[radio,out,bas,bas,dro])
                    goodsubding.click(fn=check_to_sub,inputs=[good,radio,dro,goodsubding,bas],outputs=[radio,out,bas,bas,dro])
                    badsubding.click(fn=check_to_sub,inputs=[bad,radio,dro,badsubding,bas],outputs=[radio,out,bas,bas,dro])
        with gr.Tab(label="遗失魔法碎片"):
            with gr.Row():
                with gr.Column(scale=9):
                    ttext=gr.Textbox(lines=1,label="请在此处输入中文关键词搜索可能被遗失的魔法碎片")
                    ccheck=gr.CheckboxGroup(choices=seach("",100,sf="find"),label="此处是搜索结果",value=[])
                with gr.Column(scale=1):
                    ssub = gr.Button(value="在所有tag末尾提交可能遗失的魔法碎片")
                    ssubding=gr.Button(value="在选中tag前面添加可能遗失的魔法碎片")
                    nnum = gr.Slider(minimum=1, maximum=500, step=1,value=100, label="此处是调整搜索结果个数")
            ttext.change(fn=ttext_to_check,inputs=[ttext,nnum,ccheck],outputs=[ccheck,ccheck])
            nnum.change(fn=ttext_to_check,inputs=[ttext,nnum,ccheck],outputs=[ccheck,ccheck])
            ssub.click(fn=check_to_sub,inputs=[ccheck,radio,dro,ssub,bas],outputs=[radio,out,bas,bas,dro])
            ssubding.click(fn=check_to_sub,inputs=[ccheck,radio,dro,ssubding,bas],outputs=[radio,out,bas,bas,dro])
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
            zhouradio.change(fn=zhou_to_check,inputs=zhouradio,outputs=[zhoucheck,zhoucheck,zhouout])
            zhoucheck.change(fn=zhou_to_out,inputs=[zhouradio,zhoucheck],outputs=zhouout)
            tagradio.change(fn=lambda it:getItem("tagcun",{}).get(it),inputs=tagradio,outputs=tagout)
            zhoubutton.click(fn=check_to_sub,inputs=[zhoucheck,radio,dro,zhoubutton,bas],outputs=[radio,out,bas,bas,dro])
            zhouqian.click(fn=check_to_sub,inputs=[zhoucheck,radio,dro,zhouqian,bas],outputs=[radio,out,bas,bas,dro])
            zhoudel.click(fn=zhou_del,inputs=zhouradio,outputs=[zhouradio,zhoucheck,zhoucheck,zhouout,zhouradio])
            tagbutton.click(fn=check_to_sub,inputs=[tagout,radio,dro,tagbutton,bas],outputs=[radio,out,bas,bas,dro])
            tagqian.click(fn=check_to_sub,inputs=[tagout,radio,dro,tagqian,bas],outputs=[radio,out,bas,bas,dro])
            tagdel.click(fn=tag_del,inputs=tagradio,outputs=[tagradio,tagout,tagradio])

        deleteOne.click(fn = delete_one_to_out, inputs = [radio, dro, bas], outputs = [radio, out, bas])
        delete.click(fn=delete_to_out,inputs=delete,outputs=[radio,check,out,bas,bas])
        big.click(fn=big_to_radio, inputs=radio, outputs=radio)
        small.click(fn=small_to_radio, inputs=radio, outputs=radio)
        maohao.change(fn=mao_to_radio,inputs=[radio,maohao],outputs=radio)
        radio.change(fn=radio_to_out, inputs=[dro,bas], outputs=[out,bas])
        bas.change(fn=radio_to_out,inputs=[dro,bas],outputs=[out,bas])
        # sab.change(fn=radio_to_out,inputs=[dro,bas],outputs=[out,bas])
        dro.change(fn=radio_to_out, inputs=[dro,bas], outputs=[out,bas])
        cli.click(fn=out_to_cli, inputs=out,outputs=out)
        zhoucun.click(fn=zhou_to_cun,inputs=[nname,dro,bas],outputs=[zhouradio,zhouradio])
        tagcun.click(fn=tag_to_cun,inputs=[nname,out],outputs=[tagradio,tagradio])
        
    return [(block,"maple的tag选择器","maple_tags")]
script_callbacks.on_ui_tabs(on_ui_tabs)
# on_ui_tabs()[0][0].launch()
