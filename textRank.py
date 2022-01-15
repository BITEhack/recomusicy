import spacy
import pytextrank
import pandas as pd


def transform_data(dataset, column_name, feature_num=500):
    features = {}
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("textrank")

    for index in dataset.index:
        text = dataset.at[index, column_name]
        doc = nlp(text)

        temp = []
        for phrase in doc._.phrases:
            temp.append([phrase.text, phrase.rank])
            feat = phrase.text
            rank = phrase.rank
            if feat in features:
                features[feat] += rank
            else:
                features[feat] = rank
            dataset.at[index, feat] = 1

    features_sorted = sorted([[features[x], x] for x in features.keys()], key=lambda x: x[0], reverse=True)[:feature_num]
    features_sorted = [x[1] for x in features_sorted]

    for key in features.keys():
        if key not in features_sorted:
            dataset = dataset.drop(key, axis=1)
    return dataset.fillna(0)

toxic = """Baby, can't you see I'm calling?
A guy like you should wear a warning
It's dangerous, I'm falling
There's no escape, I can't wait
I need a hit, baby, give me it
You're dangerous, I'm loving it
Too high, can't come down
Losing my head, spinnin' 'round and 'round
Do you feel me now?
With a taste of your lips, I'm on a ride
You're toxic, I'm slippin' under
With a taste of a poison paradise
I'm addicted to you
Don't you know that you're toxic?
And I love what you do
Don't you know that you're toxic?
It's getting late to give you up
I took a sip from my devil's cup
Slowly, it's taking over me
Too high, can't come down
It's in the air and it's all around
Can you feel me now?
With a taste of your lips, I'm on a ride
You're toxic, I'm slippin' under
With a taste of a poison paradise
I'm addicted to you
Don't you know that you're toxic?
And I love what you do
Don't you know that you're toxic?
Don't you know that you're toxic?
Taste of your lips, I'm on a ride
You're toxic, I'm slippin' under
With a taste of a poison paradise
I'm addicted to you
Don't you know that you're toxic?
With a taste of your lips, I'm on a ride
You're toxic, I'm slippin' under (toxic)
With a taste of a poison paradise
I'm addicted to you
Don't you know that you're toxic?
Intoxicate me now, with your lovin' now
I think I'm ready now, I think I'm ready now
Intoxicate me now, with your lovin' now
I think I'm ready now"""

random="""
This is the anthem
Told the world I need everything and some, yeah
Two girls that's a tandem
She gon' do all it for me when them bands come
Got it all, yeah, I'm young, rich, and handsome
Uh, this shit is not random
Everybody ain't got it, understand son, yeah
This shit is not random
Woah okay
It's my world and I do what I like to
I know she gon' be ready when I slide through
If you forgot I got it, I'll remind you
'Cause I got what they got I am talking times two
Disagree with me and you've been lied to
I am nothing like you, and no I'm not the type to
Follow bullshit y'all give the hype to
I can see the fakes, so when I look in your direction
Understand I'm seeing right through
You're hollow, you follow
Everybody's not the real McCoy
Worldwide, yeah, they feel the boy
California out to Georgia, Florida then up to Illinois
When I listen to your music I just feel annoyed
Haters, I don't understand them
Bred Jordan Ones from two thousand and one (OG)
Today I think I want a Phantom
If I don't get it I'ma throw a fucking tantrum, woah
This is the anthem
Told the world I need everything and some, yeah
Two girls that's a tandem
She gon' do it all for me when them bands come
Got it all, yeah, I'm young, rich, and handsome
This shit is not random (nope)
Everybody ain't got it, understand, son, yeah (sorry)
This shit is not random
Now imagine it, put the hours in and stay passionate
Wasn't blowing money, I was stacking it
Figured what the fuck I want to do in life and practiced it
Pay attention, none of this is happening by accident
Listen, I don't slack a bit
Game plan solid, no cracks in it
Said I want a billion now nothing less is adequate
Grab and check, cashing it
I was born addicted to the money
Difference is now if finally found a way to manage it
What you think I'm paid for nothing?
You must be mistaken someone
I had to go make that fund and I'm trying to be great at something
Spend not saving nothing
Flying from the bay to London
They say that money talks but you're not saying nothing
Now try and shade me, I'm like I guess
Why yes, you drive a Toyota, please define flex
Sign CDs and I sign breasts
Understand, to these female fans, I'm sex
Listen
This is the anthem
Told the world I need everything and some, yeah
Two girls that's a tandem
She gon' do it all for me when them bands come
I got it all, yeah, I'm young, rich, and handsome
This shit is not random
Everybody ain't got it, understand, son, yeah
This shit is not random"""

df = pd.DataFrame([toxic, random], columns=['text'])
df = transform_data(df, 'text', 15)
print(df)
