# Build Network
nw = nn.Network(loss_func=nn.ls.CE, batch_size=batch_size,
                lr=lr, epochs=epochs,)
l1 = nn.FCLayer(in_feature=pic_size, out_feature=num1,
                act_func=nn.act.ReLu, mean=mean, dev=dev)
l2 = nn.FCLayer(in_feature=num1, out_feature=num2,
                act_func=nn.act.ReLu, mean=mean, dev=dev)
l3 = nn.FCLayer(in_feature=num2, out_feature=char_class_num,
                act_func=nn.act.Softmax, mean=mean, dev=dev)
nw.add(l1)
nw.add(l2)
nw.add(l3)

# sys.stdout = open('output.txt', 'w')
total_time = 0
total_time = nw.classify_train(X_train, y_train, X_test, y_test)

nn.dl.save_model(nw,f'{pic_size}-{num1}-{num2}-{char_class_num}-m{mean}d{dev}-b{batch_size}-lr{lr}')