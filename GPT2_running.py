import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='review_star_1_large')


gpt2.generate(sess, run_name= 'review_star_1_large', length=50, prefix= "The environment was very dirty and", sample_delim = '<|endoftext|>', include_prefix=False, nsamples=1, return_as_list=True)
