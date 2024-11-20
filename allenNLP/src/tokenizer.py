from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from fugashi import Tagger

@Tokenizer.register("mecab")
class MecabTokenizer(Tokenizer):
    def __init__(self):
        # Taggerインスタンスを作成
        self._tagger = Tagger()

    def tokenize(self, text):
        """入力テキストをMeCabを用いて解析する"""
        tokens = []
        # 入力テキストを単語に分割
        for word in self._tagger(text):
            # 単語のテキスト（word.surface）と品詞（word.feature.pos1)からTokenインスタンスを作成
            token = Token(text=word.surface, pos_=word.feature.pos1)
            tokens.append(token)

        return tokens
