from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("../THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("../THUDM/chatglm2-6b", trust_remote_code=True).cuda()
model = model.eval()


schema = """
schema = {
    '人物': ['姓名', '性别', '出生日期', '出生地点', '职业', '获得奖项', '实体类型'],
    '书籍': ['作者', '类型', '发行时间', '定价', '实体类型'],
    '电视剧': ['导演', '演员', '题材', '出品方', '实体类型']
}
"""
context = """
1.《琅琊榜》是由山东影视传媒集团、山东影视制作有限公司、北京儒意欣欣影业投资有限公司、北京和颂天地影视文化有限公司、北京圣基影业有限公司、东阳正午阳光影视有限公司联合出品，由孔笙、李雪执导，胡歌、刘涛、王凯、黄维德、陈龙、吴磊、高鑫等主演的古装剧。
2.《满江红》是由张艺谋执导，沈腾、易烊千玺、张译、雷佳音、岳云鹏、王佳怡领衔主演，潘斌龙、余皑磊主演，郭京飞、欧豪友情出演，魏翔、张弛、黄炎特别出演，许静雅、蒋鹏宇、林博洋、飞凡、任思诺、陈永胜出演的悬疑喜剧电影。
3. 张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地男演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。
"""
ie_examples = """
{
  '人物': [
    {

  'content': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员。',

  'answers': {

    '姓名': ['岳云鹏'],

    '性别': ['男'],

    '出生日期': ['1985年4月15日'],

    '出生地点': ['河南省濮阳市南乐县'],

    '职业': ['相声演员'],

    '获得奖项': ['原文中未提及']

    }
    }
  ]
} 
"""

prompt = f"""
我定义了一个schema，schema中的每个实体具备如下属性:
{schema}
现在我有一段context, 你的任务是对其进行信息抽取,对应schema中每个实体的属性，采用Json格式输出，例如：{ie_examples}，如果原文没有，则对应字段的取值为“原文中未提及”。不要输出其它内容。
{context}
"""

response, history = model.chat(tokenizer, prompt, history=[])
print(response)