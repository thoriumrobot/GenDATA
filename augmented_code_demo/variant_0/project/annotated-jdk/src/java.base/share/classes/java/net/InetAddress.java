/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.List;
    @Positive
import java.util.NavigableSet;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Scanner;
    @Positive
import java.io.File;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectInputStream.GetField;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectOutputStream.PutField;
    @Positive
import java.lang.annotation.Native;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.concurrent.ConcurrentSkipListSet;
    @Positive
import java.util.concurrent.atomic.AtomicLong;
    @Positive
import java.util.Arrays;
    @Positive
import jdk.internal.access.JavaNetInetAddressAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.security.action.*;
    @Positive
import sun.net.InetAddressCachePolicy;
    @Positive
import sun.net.util.IPAddressUtil;
    @Positive
import sun.nio.cs.UTF_8;

    @Positive
public class InetAddress implements java.io.Serializable {

    @Positive
    static class InetAddressHolder {

    @Positive
        void init(String hostName, int family);

    @Positive
        String getHostName();

    @Positive
        String getOriginalHostName();

    @Positive
        int getAddress();

    @Positive
        int getFamily();
    @Positive
    }

    @Positive
    InetAddressHolder holder();

    @Positive
    public boolean isMulticastAddress();

    @Positive
    public boolean isAnyLocalAddress();

    @Positive
    public boolean isLoopbackAddress();

    @Positive
    public boolean isLinkLocalAddress();

    @Positive
    public boolean isSiteLocalAddress();

    @Positive
    public boolean isMCGlobal();

    @Positive
    public boolean isMCNodeLocal();

    @Positive
    public boolean isMCLinkLocal();

    @Positive
    public boolean isMCSiteLocal();

    @Positive
    public boolean isMCOrgLocal();

    @Positive
    public boolean isReachable(int timeout) throws IOException;

    @Positive
    public boolean isReachable(NetworkInterface netif, int ttl, int timeout) throws IOException;

    @Positive
    public String getHostName();

    @Positive
    String getHostName(boolean check);

    @Positive
    public String getCanonicalHostName();

    @Positive
    public byte[] getAddress();

    @Positive
    public String getHostAddress();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public String toString();

    @Positive
    private interface Addresses {

    @Positive
        InetAddress[] get() throws UnknownHostException;
    @Positive
    }

    @Positive
    private static final class CachedAddresses implements Addresses, Comparable<CachedAddresses> {

    @Positive
        @Override
    @Positive
        public InetAddress[] get() throws UnknownHostException;

    @Positive
        @Override
    @Positive
        public int compareTo(CachedAddresses other);
    @Positive
    }

    @Positive
    private static final class NameServiceAddresses implements Addresses {

    @Positive
        @Override
    @Positive
        public InetAddress[] get() throws UnknownHostException;
    @Positive
    }

    @Positive
    private interface NameService {

    @Positive
        InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;

    @Positive
        String getHostByAddr(byte[] addr) throws UnknownHostException;
    @Positive
    }

    @Positive
    private static final class PlatformNameService implements NameService {

    @Positive
        public InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;

    @Positive
        public String getHostByAddr(byte[] addr) throws UnknownHostException;
    @Positive
    }

    @Positive
    private static final class HostsFileNameService implements NameService {

    @Positive
        public HostsFileNameService(String hostsFileName) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public String getHostByAddr(byte[] addr) throws UnknownHostException;

    @Positive
        public InetAddress[] lookupAllHostAddr(String host) throws UnknownHostException;
    @Positive
    }

    @Positive
    public static InetAddress getByAddress(String host, byte[] addr) throws UnknownHostException;

    @Positive
    public static InetAddress getByName(String host) throws UnknownHostException;

    @Positive
    public static InetAddress[] getAllByName(String host) throws UnknownHostException;

    @Positive
    public static InetAddress getLoopbackAddress();

    @Positive
    static InetAddress[] getAllByName0(String host, boolean check) throws UnknownHostException;

    @Positive
    static InetAddress[] getAddressesFromNameService(String host, InetAddress reqAddr) throws UnknownHostException;

    @Positive
    public static InetAddress getByAddress(byte[] addr) throws UnknownHostException;

    @Positive
    private static final class CachedLocalHost {
    @Positive
    }

    @Positive
    public static InetAddress getLocalHost() throws UnknownHostException;

    @Positive
    static InetAddress anyLocalAddress();

    @Positive
    static InetAddressImpl loadImpl(String implName);
    @Positive
}

    @Positive
class InetAddressImplFactory {

    @Positive
    static InetAddressImpl create();

    @Positive
    static native boolean isIPv6Supported();
    @Positive
}

// CFWR semantic augmentation - variant 0
