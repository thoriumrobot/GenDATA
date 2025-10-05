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
package sun.security.krb5.internal;

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
import sun.security.krb5.Config;
    @Positive
import sun.security.krb5.Asn1Exception;
    @Positive
import sun.security.util.*;
    @Positive
import java.net.InetAddress;
    @Positive
import java.net.Inet4Address;
    @Positive
import java.net.Inet6Address;
    @Positive
import java.net.UnknownHostException;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Arrays;

    @Positive
public class HostAddress implements Cloneable {

    @Positive
    public Object clone();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public InetAddress getInetAddress() throws UnknownHostException;

    @Positive
    public HostAddress() throws UnknownHostException {
    @Positive
    }

    @Positive
    public HostAddress(int new_addrType, byte[] new_address) throws KrbApErrException, UnknownHostException {
    @Positive
    }

    @Positive
    public HostAddress(InetAddress inetAddress) {
    @Positive
    }

    @Positive
    public HostAddress(DerValue encoding) throws Asn1Exception, IOException {
    @Positive
    }

    @Positive
    public byte[] asn1Encode() throws Asn1Exception, IOException;

    @Positive
    public static HostAddress parse(DerInputStream data, byte explicitTag, boolean optional) throws Asn1Exception, IOException;

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
