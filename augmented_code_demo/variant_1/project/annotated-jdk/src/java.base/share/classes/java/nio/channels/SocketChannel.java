/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.nio.channels;

    @Positive
import org.checkerframework.checker.mustcall.qual.CreatesMustCallFor;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.mustcall.qual.NotOwning;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.net.InetSocketAddress;
    @Positive
import java.net.NetPermission;
    @Positive
import java.net.ProtocolFamily;
    @Positive
import java.net.StandardProtocolFamily;
    @Positive
import java.net.Socket;
    @Positive
import java.net.SocketOption;
    @Positive
import java.net.SocketAddress;
    @Positive
import java.net.UnixDomainSocketAddress;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.channels.spi.AbstractSelectableChannel;
    @Positive
import java.nio.channels.spi.SelectorProvider;
    @Positive
import static java.util.Objects.requireNonNull;

    @Positive
@AnnotatedFor({ "mustcall" })
    @Positive
public abstract class SocketChannel extends AbstractSelectableChannel implements ByteChannel, ScatteringByteChannel, GatheringByteChannel, NetworkChannel {

    @Positive
    protected SocketChannel(SelectorProvider provider) {
    @Positive
    }

    @Positive
    public static SocketChannel open() throws IOException;

    @Positive
    public static SocketChannel open(ProtocolFamily family) throws IOException;

    @Positive
    public static SocketChannel open(SocketAddress remote) throws IOException;

    @Positive
    public final int validOps();

    @Positive
    @Override
    @Positive
    @CreatesMustCallFor
    @Positive
    public abstract SocketChannel bind(SocketAddress local) throws IOException;

    @Positive
    @Override
    @Positive
    public abstract <T> SocketChannel setOption(SocketOption<T> name, T value) throws IOException;

    @Positive
    @NotOwning
    @Positive
    public abstract SocketChannel shutdownInput() throws IOException;

    @Positive
    @NotOwning
    @Positive
    public abstract SocketChannel shutdownOutput() throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public abstract Socket socket(@MustCallAlias SocketChannel this);

    @Positive
    public abstract boolean isConnected();

    @Positive
    public abstract boolean isConnectionPending();

    @Positive
    @CreatesMustCallFor
    @Positive
    public abstract boolean connect(SocketAddress remote) throws IOException;

    @Positive
    public abstract boolean finishConnect() throws IOException;

    @Positive
    public abstract SocketAddress getRemoteAddress() throws IOException;

    @Positive
    public abstract int read(ByteBuffer dst) throws IOException;

    @Positive
    public abstract long read(ByteBuffer[] dsts, int offset, int length) throws IOException;

    @Positive
    public final long read(ByteBuffer[] dsts) throws IOException;

    @Positive
    public abstract int write(ByteBuffer src) throws IOException;

    @Positive
    public abstract long write(ByteBuffer[] srcs, int offset, int length) throws IOException;

    @Positive
    public final long write(ByteBuffer[] srcs) throws IOException;

    @Positive
    @Override
    @Positive
    public abstract SocketAddress getLocalAddress() throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
