/*
    @Positive
 * Copyright (c) 2007, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class StandardSocketOptions {

    @Positive
    public static final SocketOption<Boolean> SO_BROADCAST;

    @Positive
    public static final SocketOption<Boolean> SO_KEEPALIVE;

    @Positive
    public static final SocketOption<Integer> SO_SNDBUF;

    @Positive
    public static final SocketOption<Integer> SO_RCVBUF;

    @Positive
    public static final SocketOption<Boolean> SO_REUSEADDR;

    @Positive
    public static final SocketOption<Boolean> SO_REUSEPORT;

    @Positive
    public static final SocketOption<Integer> SO_LINGER;

    @Positive
    public static final SocketOption<Integer> IP_TOS;

    @Positive
    public static final SocketOption<NetworkInterface> IP_MULTICAST_IF;

    @Positive
    public static final SocketOption<Integer> IP_MULTICAST_TTL;

    @Positive
    public static final SocketOption<Boolean> IP_MULTICAST_LOOP;

    @Positive
    public static final SocketOption<Boolean> TCP_NODELAY;

    @Positive
    private static class StdSocketOption<T> implements SocketOption<T> {

    @Positive
        @Override
    @Positive
        public String name();

    @Positive
        @Override
    @Positive
        public Class<T> type();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
