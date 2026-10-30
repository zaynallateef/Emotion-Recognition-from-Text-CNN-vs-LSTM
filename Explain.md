# Emotion Recognition from Text: Bilingual Study Guide

## دليل فهم مشروع التعرّف على المشاعر من النصوص

English and Arabic explanations based on the supplied CNN/LSTM project.

شرح بالإنجليزية والعربية مرتبط بالتنفيذ الفعلي للمشروع.

## Contents / المحتويات

- [01. PATTERN RECOGNITION | التعرّف على الأنماط](#topic-01)
- [02. HOW DEEP LEARNING WORKS | كيف يعمل التعلّم العميق؟](#topic-02)
- [03. TOKENS | الوحدات النصية](#topic-03)
- [04. VOCABULARY | قاموس المفردات](#topic-04)
- [05. ENCODED IDs | المعرّفات الرقمية](#topic-05)
- [06. ZERO PADDING | الحشو بالأصفار](#topic-06)
- [07. MAX LENGTH AND WHY 50 TOKENS | الطول الأقصى ولماذا 50؟](#topic-07)
- [08. EMBEDDINGS | متجهات التضمين](#topic-08)
- [09. 100 DIMENSIONS | ماذا يعني بُعد التضمين 100؟](#topic-09)
- [10. CNN AND 1D CNN | الشبكة العصبية الالتفافية أحادية البعد](#topic-10)
- [11. KERNELS AND FILTERS HERE | النواة والمرشّحات في المشروع](#topic-11)
- [12. POOLING AND GLOBAL MAX POOLING | التجميع والتجميع الأعظمي العام](#topic-12)
- [13. LSTM AND HOW IT WORKS | شبكة الذاكرة طويلة وقصيرة المدى](#topic-13)
- [14. ReLU AND DROPOUT | التفعيل والتنظيم](#topic-14)
- [15. SOFTMAX AND ARGMAX | تحويل الدرجات واختيار الفئة](#topic-15)
- [16. LOSS AND CROSS-ENTROPY | الخسارة والانتروبيا التقاطعية](#topic-16)
- [17. LOSS VERSUS ACCURACY | الفرق بين الخسارة والدقة](#topic-17)
- [18. EPOCH | دورة التدريب](#topic-18)
- [19. BATCH AND BATCH SIZE | الدفعة وحجم الدفعة](#topic-19)
- [20. ADAM AND LEARNING RATE | خوارزمية Adam ومعدل التعلّم](#topic-20)
- [21. EARLY STOPPING | الإيقاف المبكر](#topic-21)
- [22. READING SLIDE 10 | كيف نشرح صور الشريحة العاشرة؟](#topic-22)
- [23. OVERFITTING AND WHY IT APPEARS | فرط التعلّم ولماذا يحدث؟](#topic-23)
- [24. SENTENCE LENGTH DISTRIBUTION | ماذا نستفيد من توزيع أطوال الجمل؟](#topic-24)
- [25. WHY CNN DID BETTER HERE | لماذا كان CNN أفضل هنا؟](#topic-25)
- [26. CLASS IMBALANCE AND MACRO METRICS | عدم توازن الفئات](#topic-26)
- [27. SINGLE-LABEL CLASSIFICATION | التصنيف بتسمية واحدة](#topic-27)
- [28. AMBIGUITY AND IDIOMS | الغموض والتعبيرات الاصطلاحية](#topic-28)
- [29. PADDING DIFFERENCE AND RANDOM SEEDS | اختلاف الحجب والبذور العشوائية](#topic-29)
- [30. DATASET-LABEL NOISE | ضوضاء تسميات البيانات](#topic-30)
- [31. ENGLISH ONLY | التدريب على الإنجليزية فقط](#topic-31)

## PART A: THE MAIN IDEA AND TEXT REPRESENTATION

الجزء الأول: الفكرة الأساسية وتمثيل النص

<a id="topic-01"></a>

### 01. PATTERN RECOGNITION | التعرّف على الأنماط

**English**

Pattern recognition means learning how observable inputs relate to classes, then classifying new inputs. Here, the input is an English sentence and the output is one of six emotions: anger, fear, joy, love, sadness, or surprise. We represent text numerically, learn useful features, and choose a class. This is supervised learning because every training sentence has a target label.

**العربية**

التعرّف على الأنماط يعني تعلّم العلاقة بين المدخلات والفئات، ثم استخدام هذه العلاقة لتصنيف أمثلة جديدة. في مشروعنا، المُدخل جملة إنجليزية، والمُخرج إحدى ستة مشاعر: الغضب أو الخوف أو الفرح أو الحب أو الحزن أو المفاجأة. نحوّل النص إلى أرقام، ثم يتعلّم النموذج خصائص تساعده على اختيار الفئة. وهذا تعلّم خاضع للإشراف، لأن كل جملة تدريب مرتبطة بتسمية المشاعر المطلوبة.

<a id="topic-02"></a>

### 02. HOW DEEP LEARNING WORKS | كيف يعمل التعلّم العميق؟

**English**

A neural network has trainable numerical parameters called weights. During training:

1) A batch of sentences enters the network as token IDs.
2) Embeddings and CNN/LSTM layers produce learned features.
3) A final layer produces six scores, and Softmax converts them into a distribution.
4) The loss compares that distribution with the correct labels.
5) Backpropagation calculates gradients: how each parameter influences the loss.
6) Adam uses those gradients to update the weights.

Repeating this process can improve predictions. Validation checks whether improvements extend to unseen examples. During ordinary prediction, weights stay fixed and no label or weight update is needed.

**العربية**

تحتوي الشبكة العصبية على معاملات رقمية قابلة للتعلّم تُسمّى الأوزان. أثناء التدريب:

1) تدخل مجموعة من الجمل على شكل معرّفات رقمية للكلمات.
2) تحوّل طبقة التضمين هذه المعرّفات إلى متجهات، وتستخرج CNN أو LSTM خصائص منها.
3) تنتج الطبقة الأخيرة ست درجات، وتحوّلها Softmax إلى توزيع على الفئات.
4) تقارن دالة الخسارة هذا التوزيع بالتسميات الصحيحة.
5) يحسب الانتشار العكسي مقدار تأثير كل معامل في الخسارة، ويسمّى ذلك التدرّج.
6) تستخدم خوارزمية Adam التدرّجات لتحديث الأوزان.

يتكرر ذلك لتحسين التنبؤات، بينما تكشف بيانات التحقق هل يمتد التحسّن إلى أمثلة لم يتدرّب عليها النموذج. عند التنبؤ المعتاد بجملة جديدة، تبقى الأوزان ثابتة ولا نحتاج إلى تسمية صحيحة أو تحديث للأوزان.

<a id="topic-03"></a>

### 03. TOKENS | الوحدات النصية

**English**

A token is a unit produced when text is split. Depending on a tokenizer, a token could be a word, part of a word, or another unit. Our TextVectorization uses whitespace splitting, so tokens are generally words separated by spaces. Example: "i feel very happy" becomes ["i", "feel", "very", "happy"], which contains four tokens. Tokens do not automatically represent emotions.

**العربية**

الـ Token هو وحدة تنتج من تقسيم النص. قد تكون كلمة كاملة أو جزءًا من كلمة، بحسب طريقة التقسيم. في مشروعنا، تقسم TextVectorization النص بحسب المسافات، ولذلك تكون الوحدات عادةً كلمات. مثلًا، الجملة "i feel very happy" تصبح أربع وحدات: i وfeel وvery وhappy. الوحدة النصية ليست شعورًا بحد ذاتها، بل جزء من المُدخل الذي يتعلّم النموذج منه.

<a id="topic-04"></a>

### 04. VOCABULARY | قاموس المفردات

**English**

The vocabulary is the fitted lookup that connects known tokens to integer IDs. It is learned only from training text using vectorizer.adapt(...). The project caps its size at 10,000 entries, including reserved entries for padding and unknown tokens. A token not in this fitted vocabulary maps to the unknown entry. Learning vocabulary from validation or test text would leak information into preprocessing.

**العربية**

قاموس المفردات هو جدول يربط كل وحدة نصية معروفة بمعرّف رقمي. نبني هذا القاموس من نصوص التدريب فقط باستخدام vectorizer.adapt(...). الحد الأعلى في المشروع هو 10,000 مدخل، ويشمل مدخلًا للحشو وآخر للوحدات غير المعروفة. إذا ظهرت كلمة ليست في القاموس، تحصل على معرّف الوحدة المجهولة. بناء القاموس من بيانات التحقق أو الاختبار يُدخل معلومات التقييم إلى مرحلة إعداد النموذج.

<a id="topic-05"></a>

### 05. ENCODED IDs | المعرّفات الرقمية

**English**

Encoding replaces a token with its vocabulary index. In the recorded notebook, "i didnt feel humiliated" becomes [2, 139, 3, 685] before padding. These are actual recorded IDs for that fitted vocabulary. A higher ID does not mean a more positive word or a stronger emotion. IDs identify tokens; they do not measure their meaning.

Token encoding is separate from label encoding. LabelEncoder maps the six output labels as follows:
0 anger, 1 fear, 2 joy, 3 love, 4 sadness, 5 surprise.
These output-class IDs belong to a different mapping from input-token IDs.

**العربية**

الترميز الرقمي يستبدل كل وحدة نصية برقمها في القاموس. في المخرجات المسجلة للدفتر، تتحوّل "i didnt feel humiliated" إلى [2, 139, 3, 685] قبل إضافة الحشو. هذه الأرقام مرتبطة بالقاموس الذي بُني في تلك التجربة. الرقم الأكبر لا يعني أن الكلمة أكثر إيجابية أو أن الشعور أقوى. الأرقام معرّفات وليست قياسات للمعنى.

ترميز الكلمات يختلف عن ترميز تسميات المشاعر. يربط LabelEncoder فئات المخرجات بالأرقام التالية:
0 الغضب، 1 الخوف، 2 الفرح، 3 الحب، 4 الحزن، 5 المفاجأة.
هذه خريطة للفئات، وهي منفصلة عن خريطة أرقام كلمات المُدخل.

<a id="topic-06"></a>

### 06. ZERO PADDING | الحشو بالأصفار

**English**

Padding adds reserved zero IDs to short sequences so they reach the required length. A four-token sentence becomes [2, 139, 3, 685, 0, 0, ...] with 50 positions in this project. The zeros are placeholders, not real words, and not an emotion label. Padding makes examples compatible within a rectangular batch tensor.

Important distinction: token ID zero is not necessarily an all-zero embedding vector. In this project, LSTM uses mask_zero=True so padding positions do not update its recurrent state. CNN has no equivalent masking setup here, so padding embeddings can influence convolution and pooling. Padding and masking are different operations: padding adds positions; masking tells supported layers to ignore them.

**العربية**

الحشو يضيف المعرّف المحجوز صفر إلى نهاية التسلسل القصير حتى يصل إلى الطول المطلوب. فالجملة ذات الأربع وحدات تصبح [2, 139, 3, 685, 0, 0, ...] حتى يبلغ طولها 50 موضعًا. الأصفار ليست كلمات حقيقية ولا تعني فئة شعور، وإنما تشغل المواضع الفارغة لتصبح الجمل متوافقة داخل مصفوفة الدفعة.

تمييز مهم: معرّف الكلمة صفر لا يعني بالضرورة أن متجه التضمين الخاص به كله أصفار. يستخدم نموذج LSTM الإعداد mask_zero=True حتى لا تحدّث مواضع الحشو حالته المتكررة. أما CNN فلا يستخدم إعداد حجب مماثلًا هنا، ولذلك قد تؤثر متجهات الحشو في الالتفاف والتجميع. الحشو يضيف مواضع، والحجب يخبر الطبقات الداعمة بأن تتجاهل هذه المواضع.

<a id="topic-07"></a>

### 07. MAX LENGTH AND WHY 50 TOKENS | الطول الأقصى ولماذا 50؟

**English**

MAX_LENGTH = 50 sets the number of token positions in each model input. Shorter sequences receive padding. Longer sequences are truncated to the configured length. The original training data has a mean length of about 19.17 words and a 95th percentile of 41 words, so 50 retains most complete sentences while keeping inputs manageable.

Fifty is a practical design choice, not a universal rule or proof of an optimal length. Longer inputs can preserve more context but increase computation. Shorter inputs can discard useful words. This setting does not mean every original sentence has 50 real words, nor does it mean the vocabulary has only 50 words.

**العربية**

الإعداد MAX_LENGTH = 50 يحدد عدد مواضع الوحدات النصية في مُدخل كل جملة. نضيف حشوًا للجمل الأقصر، ونقتطع الجمل الأطول إلى الطول المحدد. متوسط طول جمل التدريب الأصلية يقارب 19.17 كلمة، والمئين الخامس والتسعون يساوي 41 كلمة؛ أي إن نحو 95% من الجمل طولها 41 كلمة أو أقل. لذلك يُبقي اختيار 50 معظم الجمل كاملة مع حجم إدخال مناسب للحساب.

العدد 50 قرار عملي، وليس قاعدة عامة أو دليلًا على أنه أفضل طول ممكن. زيادة الطول قد تحفظ سياقًا أكبر لكنها تزيد الحسابات، وتقليله قد يحذف كلمات مهمة. هذا الإعداد لا يعني أن كل جملة أصلية تحتوي على 50 كلمة حقيقية، ولا أن حجم القاموس 50 كلمة.

<a id="topic-08"></a>

### 08. EMBEDDINGS | متجهات التضمين

**English**

An embedding maps each token ID to a trainable numerical vector. Instead of using ID 685 as if its magnitude meant something, the model looks up a vector of 100 numbers for that token. These vectors are updated during training to support emotion classification. They are learned from this dataset, rather than imported pretrained word embeddings.

Words used in related ways may acquire useful related representations, but this is not a guarantee of human-like understanding. The embedding for a given token is a lookup vector; context is subsequently modeled by CNN or LSTM. Each of our two models learns its own embedding table.

**العربية**

طبقة التضمين تربط كل معرّف كلمة بمتجه رقمي قابل للتعلّم. بدل التعامل مع الرقم 685 على أن قيمته العددية تحمل معنى، يستخرج النموذج متجهًا من 100 رقم مرتبط بهذه الكلمة. تتغير هذه الأرقام أثناء التدريب لتساعد في تصنيف المشاعر. في مشروعنا، نتعلّمها من البيانات نفسها، ولا نستخدم متجهات كلمات جاهزة مدرّبة مسبقًا.

قد تتعلّم الكلمات المستخدمة في سياقات متقاربة تمثيلات مفيدة ومترابطة، لكن ذلك لا يضمن فهمًا شبيهًا بفهم الإنسان. متجه الكلمة في هذه الطبقة يُستخرج من جدول ثابت بعد التدريب، ثم تتعامل CNN أو LSTM مع سياق الجملة. ولكل نموذج جدول تضمين يتعلّمه بصورة مستقلة.

<a id="topic-09"></a>

### 09. 100 DIMENSIONS | ماذا يعني بُعد التضمين 100؟

**English**

output_dim=100 means each token is represented by 100 learned numerical coordinates. It does not mean 100 tokens, 100 classes, or 100 epochs. The coordinates usually have no simple individual labels such as "happiness" or "anger". Their combined pattern is useful to the model.

For one sentence: 50 token IDs become a 50 × 100 array of embedding values. For a full batch of 32 sentences, the embedding tensor has shape 32 × 50 × 100. The choice of 100 is a model hyperparameter, not a dataset statistic or a proven optimum.

**العربية**

الإعداد output_dim=100 يعني أن كل وحدة نصية تُمثّل بواسطة 100 قيمة رقمية متعلّمة. لا يعني ذلك وجود 100 كلمة أو 100 فئة أو 100 دورة تدريب. غالبًا لا نستطيع تسمية كل بُعد منفرد بأنه «فرح» أو «غضب»، بل يستفيد النموذج من نمط القيم معًا.

في الجملة الواحدة، تتحوّل المعرّفات الخمسون إلى مصفوفة حجمها 50 × 100. وفي دفعة من 32 جملة، يكون حجم موتر التضمين 32 × 50 × 100. العدد 100 معامل نختاره لتصميم النموذج، وليس إحصائية من البيانات أو قيمة ثبت أنها الأفضل.

## PART B: CNN AND LSTM

الجزء الثاني: كيف يعمل النموذجان؟

<a id="topic-10"></a>

### 10. CNN AND 1D CNN | الشبكة العصبية الالتفافية أحادية البعد

**English**

CNN stands for Convolutional Neural Network. It learns filters that detect useful local patterns. A CNN is not restricted to images. Here, Conv1D moves along one spatial axis: token position in the sentence. Each position contains a 100-dimensional embedding. "1D" describes movement along the sequence, not a one-number embedding.

Our pipeline is:
```text
50 token IDs -> Embedding(100) -> Conv1D(64, kernel size 3, ReLU)
-> GlobalMaxPooling1D -> Dropout(0.3) -> Dense(6, Softmax).
```

The convolution produces local features. Pooling keeps the strongest feature responses. The dense classifier combines them to predict an emotion. The model can use local order, although global max pooling discards the exact positions of the strongest responses.

**العربية**

اختصار CNN يعني الشبكة العصبية الالتفافية. تتعلّم مرشّحات لاكتشاف أنماط محلية مفيدة، ولا يقتصر استخدامها على الصور. في مشروعنا، تتحرك Conv1D على محور واحد هو ترتيب الكلمات في الجملة. يحتوي كل موضع على متجه تضمين من 100 بُعد. عبارة «أحادية البعد» تصف حركة المرشّح على التسلسل، ولا تعني أن تمثيل الكلمة رقم واحد.

تسلسل نموذجنا هو:
50 معرّفًا للكلمات، ثم تضمين من 100 بُعد، ثم 64 مرشّحًا بحجم نافذة 3 وتفعيل ReLU، ثم التجميع الأعظمي العام، ثم Dropout بنسبة 0.3، ثم طبقة نهائية من ست فئات مع Softmax.

يستخرج الالتفاف خصائص محلية، ويحتفظ التجميع بأقوى استجابة لكل خاصية، ثم يجمع المصنّف هذه الخصائص ليختار الشعور. يستفيد الالتفاف من ترتيب الكلمات محليًا، لكن التجميع الأعظمي العام لا يحتفظ بالموقع الدقيق للاستجابة الأقوى.

<a id="topic-11"></a>

### 11. KERNELS AND FILTERS HERE | النواة والمرشّحات في المشروع

**English**

A kernel is a learned set of weights applied repeatedly to local windows. Kernel size 3 means the window covers three adjacent token embeddings. With 100 embedding dimensions, each filter uses weights spanning 3 × 100 input values, plus a bias. It does not simply read three scalar IDs.

For "i feel very happy", successive windows include "i feel very" and "feel very happy". The same filter weights are reused at each position. Our filters=64 means 64 learned filters produce 64 feature channels. Each can learn a different useful response. We have not inspected the filters, so we cannot claim a particular filter specifically detects "joy" or "not happy".

With the notebook’s default valid convolution padding and stride 1, 50 positions become 48 positions: 50 - 3 + 1 = 48. The convolution output is therefore 48 × 64 per sentence. This convolution-padding setting is separate from the earlier zero-padding of token sequences.

**العربية**

النواة هي مجموعة أوزان متعلّمة نطبقها على نوافذ متتالية من النص. حجم النواة 3 يعني فحص ثلاث وحدات نصية متجاورة. وبما أن كل وحدة لها 100 قيمة تضمين، فإن كل مرشّح يستخدم أوزانًا تمتد على 3 × 100 قيمة، إضافةً إلى انحياز. لا يقرأ المرشّح ثلاثة معرّفات رقمية منفردة فقط.

في "i feel very happy"، تشمل النوافذ "i feel very" ثم "feel very happy". نعيد استخدام أوزان المرشّح نفسها عند كل موضع. الإعداد filters=64 يعني وجود 64 مرشّحًا متعلّمًا تنتج 64 قناة خصائص. قد يتعلّم كل منها استجابة مختلفة، لكننا لم نفحصها فعليًا، ولذلك لا يصح الجزم بأن مرشّحًا معينًا يتعرف تحديدًا على الفرح أو عبارة بعينها.

مع الإعداد الافتراضي valid وخطوة حركة واحدة، يصبح عدد المواضع 48 وفق الحساب 50 - 3 + 1. لذلك يكون خرج الالتفاف لكل جملة بحجم 48 × 64. إعداد الحشو داخل عملية الالتفاف مختلف عن حشو تسلسل الكلمات بالأصفار قبل دخوله إلى النموذج.

<a id="topic-12"></a>

### 12. POOLING AND GLOBAL MAX POOLING | التجميع والتجميع الأعظمي العام

**English**

Pooling summarizes feature responses. GlobalMaxPooling1D keeps the largest activation across all sequence positions for each filter. Illustrative example: responses [0.2, 0.8, 0.1] become 0.8. This example explains the operation; these are not measured project activations.

In our CNN, 48 × 64 convolution outputs become a 64-value vector: one maximum per filter. This reduces the representation and allows a strong phrase response to matter regardless of where it appeared. The trade-off is loss of exact position and much of the global ordering information.

**العربية**

التجميع يلخّص استجابات الخصائص. تحتفظ GlobalMaxPooling1D بأكبر تفعيل عبر جميع مواضع التسلسل لكل مرشّح. مثلًا، تتحوّل الاستجابات [0.2, 0.8, 0.1] إلى 0.8. هذا مثال توضيحي للعملية، وليس قيمًا قسناها من النموذج.

في CNN لدينا، يتحوّل الخرج من 48 × 64 إلى متجه من 64 قيمة، أي قيمة عظمى لكل مرشّح. يقلّل ذلك حجم التمثيل ويسمح للاستجابة القوية بالتأثير بغض النظر عن موضعها. المقابل هو فقدان الموقع الدقيق وكثير من المعلومات عن الترتيب العام للجملة.

<a id="topic-13"></a>

### 13. LSTM AND HOW IT WORKS | شبكة الذاكرة طويلة وقصيرة المدى

**English**

LSTM stands for Long Short-Term Memory. It is a recurrent neural network that reads a sequence step by step while updating an internal state. A cell state carries memory, and a hidden state represents information exposed at the current step.

Its learned gates regulate information:

- Forget gate: how much previous cell memory to retain.
- Input gate: how much new candidate information to add.
- Output gate: which information contributes to the hidden state.
These gates are numerical functions, not hand-written language rules. They can help model relationships across more than a short local window, but do not guarantee correct understanding of context or negation.

Our pipeline is:
```text
50 token IDs -> Embedding(100, mask_zero=True) -> LSTM(64)
-> Dropout(0.3) -> Dense(6, Softmax).
```
LSTM(64) uses a 64-value hidden representation. The layer returns its final output for sentence classification, while masking ignores padded positions. The two models are trained separately and are not a combined CNN-LSTM network.

**العربية**

اختصار LSTM يعني شبكة الذاكرة طويلة وقصيرة المدى. وهي شبكة متكررة تقرأ التسلسل خطوةً بخطوة، وتحدّث حالتها الداخلية عند كل خطوة. تحمل حالة الخلية الذاكرة، بينما تمثّل الحالة المخفية المعلومات المتاحة عند الخطوة الحالية.

تنظّم البوابات المتعلّمة تدفّق المعلومات:

- بوابة النسيان: تحدّد مقدار الذاكرة السابقة الذي نحتفظ به.
- بوابة الإدخال: تحدّد مقدار المعلومات المرشحة الجديدة التي نضيفها.
- بوابة الإخراج: تنظّم المعلومات التي تسهم في الحالة المخفية.
هذه البوابات دوال رقمية متعلّمة، وليست قواعد لغوية مكتوبة يدويًا. قد تساعد على تمثيل علاقات تتجاوز النافذة المحلية القصيرة، لكنها لا تضمن فهم السياق أو النفي بصورة صحيحة.

يبدأ نموذجنا بمعرّفات الكلمات، ثم تضمين من 100 بُعد مع mask_zero=True، ثم LSTM من 64 وحدة، ثم Dropout بنسبة 0.3، ثم ست فئات مع Softmax. تستخدم LSTM تمثيلًا مخفيًا من 64 قيمة، وتعيد خرجها النهائي لتصنيف الجملة، بينما يتجاهل الحجب مواضع الحشو. نموذج CNN ونموذج LSTM منفصلان في التدريب، ولا يشكلان شبكة واحدة مشتركة.

<a id="topic-14"></a>

### 14. ReLU AND DROPOUT | التفعيل والتنظيم

**English**

ReLU applies max(0, x), keeping positive values and replacing negative values with zero. It adds nonlinearity to CNN. Dropout(0.3) randomly suppresses 30% of its input units during training to reduce reliance on particular features. Dropout is inactive during ordinary inference. It can reduce overfitting but does not guarantee its removal.

**العربية**

تطبق ReLU العملية max(0, x)، فتحتفظ بالقيم الموجبة وتستبدل السالبة بصفر. تضيف هذه العملية علاقة غير خطية إلى CNN. أما Dropout(0.3) فيعطّل عشوائيًا 30% من وحدات مدخله أثناء التدريب لتقليل الاعتماد على خصائص بعينها. لا يحدث هذا التعطيل أثناء التنبؤ المعتاد. قد يقلّل فرط التعلّم، لكنه لا يضمن إزالته تمامًا.

<a id="topic-15"></a>

### 15. SOFTMAX AND ARGMAX | تحويل الدرجات واختيار الفئة

**English**

The final dense layer generates six real-valued scores called logits. Softmax converts them into nonnegative values that sum to one:

```text
p_k = exp(z_k) / sum_j exp(z_j)
```
```text
predicted class = argmax_k p_k
```

Softmax gives a distribution; argmax chooses the largest value. Each sentence receives one predicted label in this project. A top score near 0.25 means the winning class receives only about a quarter of the distribution. It does not demonstrate a reliable prediction. Softmax outputs have not been calibrated as confidence estimates here.

**العربية**

تنتج الطبقة الكثيفة الأخيرة ست درجات عددية تُسمّى logits. تحوّلها Softmax إلى قيم غير سالبة مجموعها واحد، وفق المعادلة أعلاه. وبعد ذلك تختار argmax الفئة ذات القيمة الأكبر.

إذًا Softmax تنتج التوزيع، وargmax تختار فئة واحدة منه. في مشروعنا، تحصل كل جملة على تسمية متوقعة واحدة. إذا كانت أعلى قيمة نحو 0.25، فهذا يعني أن الفئة الفائزة حصلت على ربع التوزيع تقريبًا، ولا يكفي ذلك لإثبات موثوقية التنبؤ. لم نختبر معايرة مخرجات Softmax بوصفها درجات ثقة في هذا المشروع.

## PART C: TRAINING, LOSS, AND THE FIGURES ON SLIDE 10

الجزء الثالث: التدريب والخسارة وقراءة رسوم الشريحة العاشرة

<a id="topic-16"></a>

### 16. LOSS AND CROSS-ENTROPY | الخسارة والانتروبيا التقاطعية

**English**

Loss is the numerical penalty used to train the model. It measures disagreement between the prediction distribution and the correct label. Lower average loss is better under the same evaluation setup.

The project uses sparse categorical cross-entropy. For one sentence:
```text
L = -ln(p_true)
```
Here, p_true is the Softmax value assigned to the correct emotion. "Sparse" means target labels are integer IDs rather than one-hot vectors; it does not mean words are removed.

Illustrative calculations:

- Correct-class score 0.9 gives loss about 0.105.
- Correct-class score 0.5 gives loss about 0.693.
- Correct-class score 0.1 gives loss about 2.303.
Confident wrong predictions can therefore contribute large loss. Batch and dataset losses summarize penalties across examples. Loss is not an error percentage, and 0.1622 loss does not mean 16.22% of sentences are wrong.

**العربية**

الخسارة هي العقوبة الرقمية التي نستخدمها لتدريب النموذج، وتعبّر عن اختلاف توزيع التنبؤ عن التسمية الصحيحة. عند المقارنة ضمن إعداد التقييم نفسه، تكون الخسارة المتوسطة الأقل أفضل.

نستخدم sparse categorical cross-entropy. خسارة الجملة الواحدة هي سالب اللوغاريتم الطبيعي للقيمة التي منحها النموذج للفئة الصحيحة. كلمة sparse تعني أن تسميات الهدف أرقام صحيحة بدل متجهات one-hot، ولا تعني حذف كلمات من النص.

أمثلة حسابية توضيحية: إذا منح النموذج الفئة الصحيحة 0.9، تكون الخسارة نحو 0.105. وعند 0.5 تصبح نحو 0.693، وعند 0.1 تصبح نحو 2.303. لذلك قد تسبب التنبؤات الخاطئة الواثقة خسارة كبيرة. تلخّص خسارة الدفعة أو مجموعة البيانات عقوبات أمثلتها. الخسارة ليست نسبة الأخطاء؛ فالقيمة 0.1622 لا تعني أن 16.22% من الجمل صُنّفت خطأ.

<a id="topic-17"></a>

### 17. LOSS VERSUS ACCURACY | الفرق بين الخسارة والدقة

**English**

Accuracy checks whether the highest-scoring class is correct. Loss also depends on the value assigned to the correct class. Two predictions can both be wrong but have very different losses. Therefore validation accuracy may stay nearly unchanged while validation loss rises as some incorrect predictions become more confident.

Final test values in this project:
CNN: accuracy 92.45%, loss 0.1622.
LSTM: accuracy 91.24%, loss 0.2626.
The lower CNN loss does not by itself establish that its confidence is calibrated.

**العربية**

تتحقق الدقة من كون الفئة ذات أعلى درجة صحيحة أم لا. أما الخسارة فتتأثر أيضًا بالقيمة التي أعطاها النموذج للفئة الصحيحة. قد يكون تنبؤان خاطئين لكن خسارتهما مختلفة جدًا. لذلك يمكن أن تبقى دقة التحقق شبه ثابتة بينما ترتفع خسارته، لأن بعض التنبؤات الخاطئة تصبح أكثر ثقة.

في الاختبار النهائي، حقق CNN دقة 92.45% وخسارة 0.1622، وحقق LSTM دقة 91.24% وخسارة 0.2626. انخفاض خسارة CNN لا يثبت وحده أن درجات ثقته معايرة جيدًا.

<a id="topic-18"></a>

### 18. EPOCH | دورة التدريب

**English**

An epoch is one pass through the training examples. Each epoch contains multiple batch updates. epochs=15 sets the maximum number of epochs, not a requirement to finish all fifteen. Early stopping can end training earlier. The supplied graphs show five completed CNN epochs and six LSTM epochs, with plot labels starting at zero.

**العربية**

الـ Epoch هو مرور كامل على أمثلة التدريب. تتضمن الدورة الواحدة تحديثات عديدة للأوزان عبر الدفعات. الإعداد epochs=15 يحدد الحد الأعلى، ولا يعني وجوب إكمال خمس عشرة دورة. قد يوقف الإيقاف المبكر التدريب قبل ذلك. تظهر الرسوم المقدّمة خمس دورات مكتملة لـ CNN وست دورات لـ LSTM، مع بدء ترقيم الرسم من الصفر.

<a id="topic-19"></a>

### 19. BATCH AND BATCH SIZE | الدفعة وحجم الدفعة

**English**

A batch is a group of examples processed together for a training update. batch_size=32 means each full training batch contains 32 sentences. With 15,969 training sentences, there are 500 batches per epoch if all examples are used without dropping the final partial batch. The last batch contains one sentence. Batch size is separate from sentence length: 32 sentences per batch, 50 token positions per sentence.

**العربية**

الدفعة هي مجموعة أمثلة نعالجها معًا لحساب تحديث تدريبي. الإعداد batch_size=32 يعني أن الدفعة الكاملة تحتوي على 32 جملة. وبوجود 15,969 جملة تدريب، نحصل على 500 دفعة في الدورة إذا استُخدمت الأمثلة كلها ولم تُحذف الدفعة الأخيرة الجزئية. تحتوي الدفعة الأخيرة على جملة واحدة. حجم الدفعة مختلف عن طول الجملة: لدينا 32 جملة في الدفعة الكاملة، ولكل جملة 50 موضعًا للوحدات النصية.

<a id="topic-20"></a>

### 20. ADAM AND LEARNING RATE | خوارزمية Adam ومعدل التعلّم

**English**

Adam is the optimizer, the algorithm that updates model parameters from gradients. It tracks moving estimates of gradients and squared gradients to adapt updates for each parameter. Our learning_rate=0.001 controls the overall update scale. It does not mean accuracy increases by 0.001 each step. Backpropagation calculates gradients; Adam uses them to change weights. The name is Adam, not "Adams".

**العربية**

Adam هي خوارزمية التحسين التي تحدّث معاملات النموذج اعتمادًا على التدرّجات. تحتفظ بتقديرات متحركة للتدرّجات ومربعاتها لتكييف التحديثات لكل معامل. يضبط learning_rate=0.001 الحجم العام للتحديث، ولا يعني زيادة الدقة بمقدار 0.001 في كل خطوة. الانتشار العكسي يحسب التدرّجات، وAdam تستخدمها لتغيير الأوزان. اسم الخوارزمية Adam.

<a id="topic-21"></a>

### 21. EARLY STOPPING | الإيقاف المبكر

**English**

Early stopping stops training after a monitored validation metric fails to improve for a configured time. Our code uses:

```python
EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
```

It monitors validation loss, tolerates three non-improving epochs, and restores the weights associated with the lowest monitored loss. It does not monitor test performance. CNN’s selected checkpoint is the second completed epoch, and LSTM’s is the third. The best validation-accuracy epoch need not be the selected epoch because checkpoint selection uses loss.

**العربية**

الإيقاف المبكر ينهي التدريب عندما يتوقف مقياس التحقق الذي نراقبه عن التحسّن لمدة محددة. يراقب الكود خسارة التحقق val_loss، ويسمح بثلاث دورات دون تحسّن، ثم يعيد الأوزان المرتبطة بأقل خسارة تحقق بفضل restore_best_weights=True. لا يستخدم أداء الاختبار لاتخاذ هذا القرار.

اختيرت أوزان الدورة الثانية المكتملة لـ CNN، والدورة الثالثة لـ LSTM. ليس ضروريًا أن تكون دورة أعلى دقة تحقق هي المختارة، لأن معيار اختيار الأوزان هنا هو الخسارة.

<a id="topic-22"></a>

### 22. READING SLIDE 10 | كيف نشرح صور الشريحة العاشرة؟

**English**

Each model’s image has two panels:

- Left: accuracy versus epoch. Higher is better.
- Right: loss versus epoch. Lower is better.
- Blue curve: training data.
- Orange curve: validation data, which does not update weights.

CNN: training accuracy rises toward 97%, but validation accuracy remains around 92%. Training loss keeps falling. Validation loss reaches its minimum of about 0.1784 at plotted index 1, which is the second completed epoch, and then rises.

LSTM: training accuracy approaches 98%, but validation accuracy stays around 91%. Training loss keeps falling. Validation loss reaches about 0.2790 at plotted index 2, which is the third completed epoch, and then worsens.

Interpretation: continued training improves fit to training data without improving validation generalization. Early stopping restores the earlier minimum-loss checkpoints. These plots show training and validation, not the final test results.

**العربية**

تحتوي صورة كل نموذج على رسمين:

- الرسم الأيسر: الدقة مقابل دورات التدريب، والأعلى أفضل.
- الرسم الأيمن: الخسارة مقابل الدورات، والأقل أفضل.
- المنحنى الأزرق: بيانات التدريب.
- المنحنى البرتقالي: بيانات التحقق، التي لا نستخدمها لتحديث الأوزان.

في CNN، ترتفع دقة التدريب نحو 97%، بينما تبقى دقة التحقق قرب 92%. تستمر خسارة التدريب في الانخفاض، لكن خسارة التحقق تصل إلى أدنى قيمة، نحو 0.1784، عند الرقم 1 على الرسم، أي الدورة الثانية المكتملة، ثم ترتفع.

في LSTM، تقترب دقة التدريب من 98%، بينما تبقى دقة التحقق قرب 91%. تنخفض خسارة التدريب، لكن خسارة التحقق تصل إلى نحو 0.2790 عند الرقم 2 على الرسم، أي الدورة الثالثة المكتملة، ثم تسوء.

التفسير: استمرار التدريب يحسّن ملاءمة أمثلة التدريب دون تحسّن مماثل على بيانات التحقق الجديدة. يعيد الإيقاف المبكر أوزان نقطة الخسارة الدنيا السابقة. هذه الرسوم تخص التدريب والتحقق، وليست نتائج الاختبار النهائي.

<a id="topic-23"></a>

### 23. OVERFITTING AND WHY IT APPEARS | فرط التعلّم ولماذا يحدث؟

**English**

Overfitting means fitting training-specific details that do not generalize well. The growing training/validation gap and rising validation loss after an early minimum are evidence of it here. A network has many trainable parameters and can increasingly fit details of a finite training set as optimization continues.

Limited data diversity, ambiguous labels, and model capacity may contribute, but the graphs alone do not prove the exact cause. Do not claim that LSTM inherently overfits or that class imbalance alone caused this pattern. Possible future responses include tuning regularization or dropout, using smaller models, and collecting better data. Compare changes using validation data and reserve the test set for final assessment.

**العربية**

فرط التعلّم يعني تعلّم تفاصيل خاصة بأمثلة التدريب لا تنتقل جيدًا إلى بيانات جديدة. اتساع الفجوة بين التدريب والتحقق، وارتفاع خسارة التحقق بعد حد أدنى مبكر، دليلان عليه في تجربتنا. تحتوي الشبكة على معاملات كثيرة، وقد تستمر في ملاءمة تفاصيل مجموعة التدريب المحدودة كلما واصلنا التحسين.

قد تسهم محدودية تنوع البيانات أو غموض التسميات أو سعة النموذج، لكن الرسوم لا تثبت السبب الدقيق. لا يصح القول إن LSTM يفرط في التعلّم بطبيعته، أو إن عدم توازن الفئات وحده سبب ذلك. يمكن مستقبلًا ضبط التنظيم وDropout، أو تجربة نماذج أصغر، أو جمع بيانات أفضل. نقارن التعديلات باستخدام بيانات التحقق، ونحافظ على الاختبار للتقييم النهائي.

## PART D: DATA INSIGHTS AND MODEL COMPARISON

الجزء الرابع: فهم البيانات ومقارنة النتائج

<a id="topic-24"></a>

### 24. SENTENCE LENGTH DISTRIBUTION | ماذا نستفيد من توزيع أطوال الجمل؟

**English**

The horizontal axis shows word count and the vertical axis shows frequency. The figure summarizes the original 16,000 training sentences before the later deduplication step. Their mean is about 19.17 words, median 17, 95th percentile 41, and maximum 66. The distribution has a tail toward longer sentences.

It helps us choose input length, anticipate padding and truncation, and estimate computation. A length of 50 exceeds the 95th percentile, so most original sentences fit completely. The figure does not provide the exact percentage exceeding 50 without counting those cases. Short texts also motivate investigating local CNN features, but this histogram does not prove why CNN outperformed LSTM.

**العربية**

يمثّل المحور الأفقي عدد الكلمات، والمحور العمودي تكرار الجمل. يلخّص الرسم جمل التدريب الأصلية البالغ عددها 16,000 قبل خطوة إزالة التكرار اللاحقة. متوسط الطول نحو 19.17 كلمة، والوسيط 17، والمئين الخامس والتسعون 41، وأقصى طول 66. يمتد ذيل التوزيع باتجاه الجمل الأطول.

يساعدنا ذلك على اختيار طول المُدخل، وتوقّع مقدار الحاجة إلى الحشو أو الاقتطاع، وتقدير الحسابات. يتجاوز الطول 50 المئين الخامس والتسعين، ولذلك تتسع المدخلات لمعظم الجمل الأصلية كاملة. لا يعطينا الرسم النسبة الدقيقة للجمل التي تتجاوز 50 دون عدّها. كما أن قِصر النصوص يدعم دراسة الخصائص المحلية لـ CNN، لكنه لا يثبت سبب تفوقه.

<a id="topic-25"></a>

### 25. WHY CNN DID BETTER HERE | لماذا كان CNN أفضل هنا؟

**English**

The observed result is that CNN performs better overall in this run:

- Accuracy: CNN 92.45%, LSTM 91.24%.
- Macro F1: CNN 0.8760, LSTM 0.8666.
- Correct predictions: CNN 1,836, LSTM 1,812, out of 1,986.
- Macro recall: CNN 0.8844, LSTM 0.8851, a small LSTM advantage.

A plausible explanation is that many sentences are short and local emotional expressions are informative. CNN directly learns such local patterns. The extra sequential modeling of LSTM did not improve overall scores in this experiment. This is an interpretation, not a measured causal finding. Padding differs between models and training was not repeated across multiple seeds.

LSTM recognizes 52 of 65 surprise examples versus CNN’s 46, giving surprise recall of 80% versus about 70.8%. CNN is therefore the better overall choice for this recorded setup, but does not win every metric or every class.

**العربية**

النتيجة المرصودة هي تفوق CNN إجمالًا في هذه التجربة: دقته 92.45% مقابل 91.24%، وMacro F1 يساوي 0.8760 مقابل 0.8666. أصاب CNN في 1,836 جملة، وأصاب LSTM في 1,812 من أصل 1,986. في المقابل، يتقدم LSTM بفارق صغير في Macro Recall، إذ حقق 0.8851 مقابل 0.8844.

التفسير المحتمل هو أن كثيرًا من الجمل قصير، وأن التعبيرات العاطفية المحلية تحمل معلومات مفيدة يلتقطها CNN مباشرةً. لم تضف النمذجة التسلسلية في LSTM تحسّنًا إجماليًا في هذه التجربة. هذا تفسير وليس نتيجة سببية أثبتناها. كما تختلف معالجة الحشو، ولم نكرر التدريب عبر بذور عشوائية متعددة.

تعرّف LSTM على 52 من 65 مثال مفاجأة، مقابل 46 لـ CNN، أي استدعاء 80% مقابل نحو 70.8%. لذلك CNN هو الخيار الأفضل إجمالًا لهذا الإعداد المسجل، لكنه لا يتفوّق في كل مقياس أو فئة.

<a id="topic-26"></a>

### 26. CLASS IMBALANCE AND MACRO METRICS | عدم توازن الفئات

**English**

Class imbalance means some labels have many more examples than others. Original training includes 5,362 joy sentences but only 572 surprise sentences. A model can achieve high overall accuracy while doing poorly on rare classes. Macro metrics average class scores equally, so frequent classes do not dominate the average by sample count.

Future options include class weights, collecting rare-class data, or balanced training sampling. Class weights increase the training penalty associated with selected classes. These methods were not implemented in the recorded comparison. Keep the test set unchanged when comparing with the reported results. Balancing test data would change the evaluation population and reduce direct comparability.

**العربية**

عدم توازن الفئات يعني أن بعض التسميات لديها أمثلة أكثر بكثير من غيرها. مثلًا، تحتوي بيانات التدريب الأصلية على 5,362 جملة فرح مقابل 572 جملة مفاجأة فقط. قد يحقق النموذج دقة إجمالية عالية مع أداء ضعيف على الفئات النادرة. تعطي مقاييس Macro الفئات وزنًا متساويًا عند حساب المتوسط، فلا تهيمن الفئة الكبيرة بسبب عدد أمثلتها وحده.

يمكن مستقبلًا استخدام أوزان للفئات، أو جمع أمثلة إضافية للفئات النادرة، أو موازنة اختيار أمثلة التدريب. تزيد أوزان الفئات عقوبة التدريب المرتبطة بفئات مختارة. لم نطبق هذه الطرق في المقارنة المسجلة. يجب إبقاء مجموعة الاختبار كما هي عند المقارنة بالنتائج الحالية؛ لأن موازنتها تغيّر طبيعة الأمثلة التي نقيس الأداء عليها.

## PART E: LIMITATIONS AND WHAT THEY MEAN

الجزء الخامس: القيود ومعناها في المشروع

<a id="topic-27"></a>

### 27. SINGLE-LABEL CLASSIFICATION | التصنيف بتسمية واحدة

**English**

Each dataset sentence has one target emotion, and argmax returns one prediction. Real sentences can express several emotions at once, such as happiness about success and sadness about leaving friends. Our current setup cannot explicitly assign multiple labels. A multi-label task would need suitable annotations and a different output/loss setup, rather than simply keeping this six-way Softmax classifier unchanged.

**العربية**

لكل جملة في البيانات شعور مستهدف واحد، وتعيد argmax تنبؤًا واحدًا. لكن الجملة الحقيقية قد تجمع مشاعر متعددة، مثل الفرح بالنجاح والحزن على فراق الأصدقاء. لا يسمح إعدادنا الحالي بإسناد تسميات متعددة صراحةً. يتطلب ذلك بيانات موسومة بما يناسب المهمة، وإعدادًا مختلفًا للمخرجات والخسارة، بدل إبقاء مصنّف Softmax الحالي كما هو.

<a id="topic-28"></a>

### 28. AMBIGUITY AND IDIOMS | الغموض والتعبيرات الاصطلاحية

**English**

Ambiguity occurs when text supports several meanings or lacks enough context. An idiom has a meaning that is not obvious from individual words. "I feel so blue today" normally expresses sadness, rather than a color. CNN selects sadness here with a top score around 0.25 and joy close behind. That is an uncertain distribution even though the selected label is plausible.

For "I never expected this to happen", CNN predicts joy at 0.239 and LSTM anger at 0.376. Missing positive or negative context makes a definitive reading difficult. Limited phrase coverage or vocabulary could contribute, but the notebook does not isolate the specific cause of each prediction.

**العربية**

يظهر الغموض عندما يحتمل النص معاني متعددة أو يفتقر إلى السياق الكافي. أما التعبير الاصطلاحي فمعناه لا يُستنتج مباشرةً من كلماته المنفردة. تعني عبارة "I feel so blue today" عادةً الحزن، لا اللون الأزرق. اختار CNN الحزن بدرجة عليا نحو 0.25، مع اقتراب الفرح منه. هذا توزيع غير حاسم حتى لو كانت التسمية المختارة معقولة.

في "I never expected this to happen"، اختار CNN الفرح بدرجة 0.239 واختار LSTM الغضب بدرجة 0.376. غياب سياق إيجابي أو سلبي يجعل الجزم صعبًا. قد تسهم محدودية تغطية العبارات أو القاموس، لكن الدفتر لا يحدد سبب كل تنبؤ بصورة مستقلة.

<a id="topic-29"></a>

### 29. PADDING DIFFERENCE AND RANDOM SEEDS | اختلاف الحجب والبذور العشوائية

**English**

Both models receive 50-position sequences, but LSTM masks zero IDs and CNN does not use the same masking setup. This compares the implemented pipelines, not a perfectly isolated architecture change. A future experiment could control padding behavior more carefully.

A random seed controls parts of the random process, such as weight initialization and data shuffling. The notebook uses seed 42. One seed does not show how stable the result is. Repeating several seeds can estimate variability and show whether the ranking is consistent. It does not require repeatedly tuning on test results.

**العربية**

يستقبل النموذجان تسلسلات من 50 موضعًا، لكن LSTM يحجب المعرّفات الصفرية، بينما لا يستخدم CNN إعداد الحجب نفسه. لذلك نقارن مساري التنفيذ الفعليين، ولا نعزل اختلاف البنية وحده بصورة مثالية. يمكن في تجربة مستقبلية ضبط معالجة الحشو بعناية أكبر.

تتحكم البذرة العشوائية في أجزاء من العمليات العشوائية، مثل تهيئة الأوزان وترتيب البيانات. يستخدم الدفتر البذرة 42، لكن تجربة بذرة واحدة لا تكشف استقرار النتيجة. يسمح تكرار بذور متعددة بتقدير التباين ومعرفة هل يستمر ترتيب النموذجين. ولا يعني ذلك استخدام نتائج الاختبار مرارًا لضبط النموذج.

<a id="topic-30"></a>

### 30. DATASET-LABEL NOISE | ضوضاء تسميات البيانات

**English**

Some examples may have questionable labels or support more than one reasonable emotion. A disagreement with the dataset is not always proof that the model completely misunderstood the text, but we should not automatically assume the dataset is wrong either. Inspect examples carefully and distinguish annotation uncertainty from model errors.

Use validation data for repeated model selection. Repeatedly changing the model after examining test mistakes gradually turns the test set into a tuning resource and weakens its independence.

**العربية**

قد تكون بعض التسميات قابلة للنقاش، أو قد تعبّر الجملة عن أكثر من شعور معقول. اختلاف النموذج عن تسمية البيانات ليس دائمًا دليلًا على سوء فهم كامل، لكنه لا يعني أيضًا أن البيانات خاطئة تلقائيًا. نفحص الأمثلة بعناية ونميّز عدم اليقين في الوسم من أخطاء النموذج.

نستخدم بيانات التحقق لاختيار النموذج بصورة متكررة. إذا غيّرنا النموذج مرارًا بناءً على أخطاء الاختبار، تصبح مجموعة الاختبار عمليًا أداة ضبط وتضعف استقلاليتها في التقييم النهائي.

<a id="topic-31"></a>

### 31. ENGLISH ONLY | التدريب على الإنجليزية فقط

**English**

The vocabulary and models were trained on the supplied English-language data. Performance on Arabic or other languages has not been measured. Arabic text may map largely to unknown tokens under this vocabulary. Translating the explanation into Arabic does not make the model an Arabic emotion classifier. Supporting Arabic would require suitable data and preprocessing, followed by actual evaluation.

**العربية**

تدرّب القاموس والنموذجان على البيانات الإنجليزية المقدّمة. لم نقس الأداء على العربية أو لغات أخرى. قد تتحوّل كلمات عربية كثيرة إلى المعرّف المجهول داخل هذا القاموس. ترجمة شرح المشروع إلى العربية لا تجعل النموذج مصنّف مشاعر عربيًا. يحتاج دعم العربية إلى بيانات ومعالجة مناسبة، ثم تقييم فعلي للأداء.

## QUICK REVISION: DO NOT CONFUSE THESE NUMBERS

مراجعة سريعة: لا تخلط بين هذه الأعداد

- 10,000 = maximum vocabulary entries, including reserved entries.
- 50 = token positions per sentence after padding or truncation.
- 100 = embedding coordinates per token.
- 3 = adjacent token positions covered by each convolution window.
- 64 CNN filters = learned local feature channels.
- 64 LSTM units = size of its hidden representation.
- 32 = sentences in a full training batch.
- 15 = maximum training epochs.
- 3 early-stopping patience = non-improving epochs tolerated.
- 0.001 = Adam learning rate.
- 0.3 = dropout rate during training.
- 6 = output emotion classes.

- 10,000 = الحد الأعلى لمدخلات القاموس، بما فيها المدخلات المحجوزة.
- 50 = مواضع الوحدات النصية لكل جملة بعد الحشو أو الاقتطاع.
- 100 = عدد قيم التضمين لكل وحدة نصية.
- 3 = عدد المواضع المتجاورة داخل نافذة الالتفاف.
- 64 مرشّحًا في CNN = عدد قنوات الخصائص المحلية المتعلّمة.
- 64 وحدة في LSTM = حجم التمثيل المخفي.
- 32 = عدد الجمل في دفعة التدريب الكاملة.
- 15 = الحد الأعلى لدورات التدريب.
- 3 في patience = عدد دورات عدم التحسّن المسموح بها.
- 0.001 = معدل تعلّم Adam.
- 0.3 = نسبة Dropout أثناء التدريب.
- 6 = عدد فئات المشاعر في المخرجات.

PROJECT SOURCES | مصادر معلومات المشروع

The explanations use the supplied code and recorded outputs, not new training runs.
يعتمد الشرح على الكود والمخرجات المسجلة المقدّمة، وليس على تجارب تدريب جديدة.

- D:/pattern_recognition_project/code.ipynb
- D:/pattern_recognition_project/README.md
- D:/pattern_recognition_project/results/model_comparison.csv
- Supplied class-distribution, sentence-length, learning-curve,
  confusion-matrix, and custom-prediction figures in results/.

Code locations, using zero-based notebook cell indices:
Cleaning 2; EDA 3 and 5; deduplication 7; label encoding 11;
vectorization 13–14; CNN 19–23; LSTM 26–31;
test evaluation 33–37; new-sentence predictions 39–43.

Dataset attribution: Kaggle, Emotions Dataset for NLP.
[Kaggle: Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
