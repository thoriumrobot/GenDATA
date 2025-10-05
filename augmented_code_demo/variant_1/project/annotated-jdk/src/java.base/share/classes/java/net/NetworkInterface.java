/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
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
import java.util.Arrays;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;

    @Positive
public final class NetworkInterface {

    @Positive
    public String getName();

    @Positive
    public Enumeration<InetAddress> getInetAddresses();

    @Positive
    public Stream<InetAddress> inetAddresses();

    @Positive
    public java.util.List<InterfaceAddress> getInterfaceAddresses();

    @Positive
    public Enumeration<NetworkInterface> getSubInterfaces();

    @Positive
    public Stream<NetworkInterface> subInterfaces();

    @Positive
    public NetworkInterface getParent();

    @Positive
    public int getIndex();

    @Positive
    public String getDisplayName();

    @Positive
    public static NetworkInterface getByName(String name) throws SocketException;

    @Positive
    public static NetworkInterface getByIndex(int index) throws SocketException;

    @Positive
    public static NetworkInterface getByInetAddress(InetAddress addr) throws SocketException;

    @Positive
    public static Enumeration<NetworkInterface> getNetworkInterfaces() throws SocketException;

    @Positive
    public static Stream<NetworkInterface> networkInterfaces() throws SocketException;

    @Positive
    static boolean isBoundInetAddress(InetAddress addr) throws SocketException;

    @Positive
    public boolean isUp() throws SocketException;

    @Positive
    public boolean isLoopback() throws SocketException;

    @Positive
    public boolean isPointToPoint() throws SocketException;

    @Positive
    public boolean supportsMulticast() throws SocketException;

    @Positive
    public byte[] getHardwareAddress() throws SocketException;

    @Positive
    public int getMTU() throws SocketException;

    @Positive
    public boolean isVirtual();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public String toString();

    @Positive
    static NetworkInterface getDefault();
    @Positive
}

// CFWR semantic augmentation - variant 1
