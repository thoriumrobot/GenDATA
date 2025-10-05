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
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.NetPermission;
import java.net.ProtocolFamily;
import java.net.ServerSocket;
import java.net.SocketOption;
import java.net.SocketAddress;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.spi.AbstractSelectableChannel;
import java.nio.channels.spi.SelectorProvider;
import static java.util.Objects.requireNonNull;

@AnnotatedFor({ "mustcall" })
public abstract class ServerSocketChannel extends AbstractSelectableChannel implements NetworkChannel {

    protected ServerSocketChannel(SelectorProvider provider) {
    }

    public static ServerSocketChannel open() throws IOException;

    public static ServerSocketChannel open(ProtocolFamily family) throws IOException;

    public final int validOps();

    @CreatesMustCallFor
    @MustCallAlias
    public final ServerSocketChannel bind(@MustCallAlias ServerSocketChannel this, SocketAddress local) throws IOException;

    @CreatesMustCallFor
    @MustCallAlias
    public abstract ServerSocketChannel bind(@MustCallAlias ServerSocketChannel this, SocketAddress local, int backlog) throws IOException;

    public abstract <T> ServerSocketChannel setOption(SocketOption<T> name, T value) throws IOException;

    @MustCallAlias
    public abstract ServerSocket socket(@MustCallAlias ServerSocketChannel this);

    @MustCallAlias
    public abstract SocketChannel accept(@MustCallAlias ServerSocketChannel this) throws IOException;

    @Override
    public abstract SocketAddress getLocalAddress() throws IOException;
}
