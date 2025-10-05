/*
 * Copyright (c) 2007, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.nio.channels;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.channels.spi.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Future;
import java.io.IOException;
import java.net.SocketOption;
import java.net.SocketAddress;
import java.nio.ByteBuffer;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class AsynchronousSocketChannel implements AsynchronousByteChannel, NetworkChannel {

    protected AsynchronousSocketChannel(AsynchronousChannelProvider provider) {
    }

    public final AsynchronousChannelProvider provider();

    public static AsynchronousSocketChannel open(AsynchronousChannelGroup group) throws IOException;

    public static AsynchronousSocketChannel open() throws IOException;

    @Override
    public abstract AsynchronousSocketChannel bind(SocketAddress local) throws IOException;

    @Override
    public abstract <T> AsynchronousSocketChannel setOption(SocketOption<T> name, T value) throws IOException;

    public abstract AsynchronousSocketChannel shutdownInput() throws IOException;

    public abstract AsynchronousSocketChannel shutdownOutput() throws IOException;

    public abstract SocketAddress getRemoteAddress() throws IOException;

    public abstract <A> void connect(SocketAddress remote, A attachment, CompletionHandler<Void, ? super A> handler);

    public abstract Future<Void> connect(SocketAddress remote);

    public abstract <A> void read(ByteBuffer dst, long timeout, TimeUnit unit, A attachment, CompletionHandler<Integer, ? super A> handler);

    @Override
    public final <A> void read(ByteBuffer dst, A attachment, CompletionHandler<Integer, ? super A> handler);

    @Override
    public abstract Future<Integer> read(ByteBuffer dst);

    public abstract <A> void read(ByteBuffer[] dsts, int offset, int length, long timeout, TimeUnit unit, A attachment, CompletionHandler<Long, ? super A> handler);

    public abstract <A> void write(ByteBuffer src, long timeout, TimeUnit unit, A attachment, CompletionHandler<Integer, ? super A> handler);

    @Override
    public final <A> void write(ByteBuffer src, A attachment, CompletionHandler<Integer, ? super A> handler);

    @Override
    public abstract Future<Integer> write(ByteBuffer src);

    public abstract <A> void write(ByteBuffer[] srcs, int offset, int length, long timeout, TimeUnit unit, A attachment, CompletionHandler<Long, ? super A> handler);

    public abstract SocketAddress getLocalAddress() throws IOException;
}
