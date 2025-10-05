/*
 * Copyright (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.net;

import org.checkerframework.checker.signedness.qual.SignedPositive;
import java.lang.annotation.Native;

public interface SocketOptions {

    public void setOption(int optID, Object value) throws SocketException;

    public Object getOption(int optID) throws SocketException;

    @Native
    @SignedPositive
    public static final int TCP_NODELAY;

    @Native
    @SignedPositive
    public static final int SO_BINDADDR;

    @Native
    @SignedPositive
    public static final int SO_REUSEADDR;

    @Native
    @SignedPositive
    public static final int SO_REUSEPORT;

    @Native
    @SignedPositive
    public static final int SO_BROADCAST;

    @Native
    @SignedPositive
    public static final int IP_MULTICAST_IF;

    @Native
    @SignedPositive
    public static final int IP_MULTICAST_IF2;

    @Native
    @SignedPositive
    public static final int IP_MULTICAST_LOOP;

    @Native
    @SignedPositive
    public static final int IP_TOS;

    @Native
    @SignedPositive
    public static final int SO_LINGER;

    @Native
    @SignedPositive
    public static final int SO_TIMEOUT;

    @Native
    @SignedPositive
    public static final int SO_SNDBUF;

    @Native
    @SignedPositive
    public static final int SO_RCVBUF;

    @Native
    @SignedPositive
    public static final int SO_KEEPALIVE;

    @Native
    @SignedPositive
    public static final int SO_OOBINLINE;
}
