for cocoid in annos:
   n = len(annos[cocoid]['final_captions'])
   input_seq = np.zeros((n, args.max_length + 1), dtype='uint32')
   target_seq = np.zeros((n, args.max_length + 1), dtype='int32') - 1
   for j,s in enumerate(annos[cocoid]['final_captions']):
      for k,w in enumerate(s):
         if k < args.max_length:
            input_seq[j,k+1] = wtoi[w]
            target_seq[j,k] = wtoi[w]
            
      seq_len = len(s)
      if seq_len <= args.max_length:
            target_seq[j,seq_len] = 0
      else:
            target_seq[j,args.max_length] = wtoi[s[args.max_length]]
