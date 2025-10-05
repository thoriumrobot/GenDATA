/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.mustcall.qual.CreatesMustCallFor;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.mustcall.qual.NotOwning;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.NetPermission;
import java.net.ProtocolFamily;
import java.net.StandardProtocolFamily;
import java.net.Socket;
import java.net.SocketOption;
import java.net.SocketAddress;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.spi.AbstractSelectableChannel;
import java.nio.channels.spi.SelectorProvider;
import static java.util.Objects.requireNonNull;

@AnnotatedFor({ "mustcall" })
public abstract class SocketChannel extends AbstractSelectableChannel implements ByteChannel, ScatteringByteChannel, GatheringByteChannel, NetworkChannel {

    protected SocketChannel(SelectorProvider provider) {
    }

    public static SocketChannel open() throws IOException;

    public static SocketChannel open(ProtocolFamily family) throws IOException;

    public static SocketChannel open(SocketAddress remote) throws IOException;

    public final int validOps();

    @Override
    @CreatesMustCallFor
    public abstract SocketChannel bind(SocketAddress local) throws IOException;

    @Override
    public abstract <T> SocketChannel setOption(SocketOption<T> name, T value) throws IOException;

    @NotOwning
    public abstract SocketChannel shutdownInput() throws IOException;

    @NotOwning
    public abstract SocketChannel shutdownOutput() throws IOException;

    @MustCallAlias
    public abstract Socket socket(@MustCallAlias SocketChannel this);

    public abstract boolean isConnected();

    public abstract boolean isConnectionPending();

    @CreatesMustCallFor
    public abstract boolean connect(SocketAddress remote) throws IOException;

    public abstract boolean finishConnect() throws IOException;

    public abstract SocketAddress getRemoteAddress() throws IOException;

    public abstract int read(ByteBuffer dst) throws IOException;

    public abstract long read(ByteBuffer[] dsts, int offset, int length) throws IOException;

    public final long read(ByteBuffer[] dsts) throws IOException;

    public abstract int write(ByteBuffer src) throws IOException;

    public abstract long write(ByteBuffer[] srcs, int offset, int length) throws IOException;

    public final long write(ByteBuffer[] srcs) throws IOException;

    @Override
    public abstract SocketAddress getLocalAddress() throws IOException;
}
