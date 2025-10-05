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
import java.net.SocketOption;
import java.net.SocketAddress;
import java.util.concurrent.Future;
import java.io.IOException;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class AsynchronousServerSocketChannel implements AsynchronousChannel, NetworkChannel {

    protected AsynchronousServerSocketChannel(AsynchronousChannelProvider provider) {
    }

    public final AsynchronousChannelProvider provider();

    public static AsynchronousServerSocketChannel open(AsynchronousChannelGroup group) throws IOException;

    public static AsynchronousServerSocketChannel open() throws IOException;

    public final AsynchronousServerSocketChannel bind(SocketAddress local) throws IOException;

    public abstract AsynchronousServerSocketChannel bind(SocketAddress local, int backlog) throws IOException;

    public abstract <T> AsynchronousServerSocketChannel setOption(SocketOption<T> name, T value) throws IOException;

    public abstract <A> void accept(A attachment, CompletionHandler<AsynchronousSocketChannel, ? super A> handler);

    public abstract Future<AsynchronousSocketChannel> accept();

    @Override
    public abstract SocketAddress getLocalAddress() throws IOException;
}
