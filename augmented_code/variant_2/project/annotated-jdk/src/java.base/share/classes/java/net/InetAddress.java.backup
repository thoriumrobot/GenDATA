/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.util.List;
import java.util.NavigableSet;
import java.util.ArrayList;
import java.util.Objects;
import java.util.Scanner;
import java.io.File;
import java.io.ObjectStreamException;
import java.io.ObjectStreamField;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectInputStream.GetField;
import java.io.ObjectOutputStream;
import java.io.ObjectOutputStream.PutField;
import java.lang.annotation.Native;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.atomic.AtomicLong;
import java.util.Arrays;
import jdk.internal.access.JavaNetInetAddressAccess;
import jdk.internal.access.SharedSecrets;
import sun.security.action.*;
import sun.net.InetAddressCachePolicy;
import sun.net.util.IPAddressUtil;
import sun.nio.cs.UTF_8;

public class InetAddress implements java.io.Serializable {

    static class InetAddressHolder {

        void init(String hostName, int family);

        String getHostName();

        String getOriginalHostName();

        int getAddress();

        int getFamily();
    }

    InetAddressHolder holder();

    public boolean isMulticastAddress();

    public boolean isAnyLocalAddress();

    public boolean isLoopbackAddress();

    public boolean isLinkLocalAddress();

    public boolean isSiteLocalAddress();

    public boolean isMCGlobal();

    public boolean isMCNodeLocal();

    public boolean isMCLinkLocal();

    public boolean isMCSiteLocal();

    public boolean isMCOrgLocal();

    public boolean isReachable(int timeout) throws IOException;

    public boolean isReachable(NetworkInterface netif, int ttl, int timeout) throws IOException;

    public String getHostName();

    String getHostName(boolean check);

    public String getCanonicalHostName();

    public byte[] getAddress();

    public String getHostAddress();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public String toString();

    private interface Addresses {

        InetAddress[] get() throws UnknownHostException;
    }

    private static final class CachedAddresses implements Addresses, Comparable<CachedAddresses> {

        @Override
        public InetAddress[] get() throws UnknownHostException;

        @Override
        public int compareTo(CachedAddresses other);
    }

    private static final class NameServiceAddresses implements Addresses {

        @Override
        public InetAddress[] get() throws UnknownHostException;
    }

    private interface NameService {

        InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;

        String getHostByAddr(byte[] addr) throws UnknownHostException;
    }

    private static final class PlatformNameService implements NameService {

        public InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;

        public String getHostByAddr(byte[] addr) throws UnknownHostException;
    }

    private static final class HostsFileNameService implements NameService {

        public HostsFileNameService(String hostsFileName) {
        }

        @Override
        public String getHostByAddr(byte[] addr) throws UnknownHostException;

        public InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;
    }

    public static InetAddress getByAddress(String host, byte[] addr) throws UnknownHostException;

    public static InetAddress getByName(String host) throws UnknownHostException;

    public static InetAddress[] getAllByName(String host) throws UnknownHostException;

    public static InetAddress getLoopbackAddress();

    static InetAddress[] getAllByName0(String host, boolean check) throws UnknownHostException;

    static InetAddress[] getAddressesFromNameService(String host, InetAddress reqAddr) throws UnknownHostException;

    public static InetAddress getByAddress(byte[] addr) throws UnknownHostException;

    private static final class CachedLocalHost {
    }

    public static InetAddress getLocalHost() throws UnknownHostException;

    static InetAddress anyLocalAddress();

    static InetAddressImpl loadImpl(String implName);
}

class InetAddressImplFactory {

    static InetAddressImpl create();

    static native boolean isIPv6Supported();
}
