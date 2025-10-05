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
package javax.security.auth.kerberos;

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
import java.io.*;
    @Positive
import java.util.Date;
    @Positive
import java.util.Arrays;
    @Positive
import java.net.InetAddress;
    @Positive
import java.util.Objects;
    @Positive
import javax.crypto.SecretKey;
    @Positive
import javax.security.auth.Refreshable;
    @Positive
import javax.security.auth.Destroyable;
    @Positive
import javax.security.auth.RefreshFailedException;
    @Positive
import javax.security.auth.DestroyFailedException;
    @Positive
import sun.security.util.HexDumpEncoder;

    @Positive
public class KerberosTicket implements Destroyable, Refreshable, java.io.Serializable {

    @Positive
    public KerberosTicket(byte[] asn1Encoding, KerberosPrincipal client, KerberosPrincipal server, byte[] sessionKey, int keyType, boolean[] flags, Date authTime, Date startTime, Date endTime, Date renewTill, InetAddress[] clientAddresses) {
    @Positive
    }

    @Positive
    public final KerberosPrincipal getClient();

    @Positive
    public final KerberosPrincipal getServer();

    @Positive
    public final SecretKey getSessionKey();

    @Positive
    public final int getSessionKeyType();

    @Positive
    public final boolean isForwardable();

    @Positive
    public final boolean isForwarded();

    @Positive
    public final boolean isProxiable();

    @Positive
    public final boolean isProxy();

    @Positive
    public final boolean isPostdated();

    @Positive
    public final boolean isRenewable();

    @Positive
    public final boolean isInitial();

    @Positive
    public final boolean[] getFlags();

    @Positive
    public final java.util.Date getAuthTime();

    @Positive
    public final java.util.Date getStartTime();

    @Positive
    public final java.util.Date getEndTime();

    @Positive
    public final java.util.Date getRenewTill();

    @Positive
    public final java.net.InetAddress[] getClientAddresses();

    @Positive
    public final byte[] getEncoded();

    @Positive
    public boolean isCurrent();

    @Positive
    public void refresh() throws RefreshFailedException;

    @Positive
    public void destroy() throws DestroyFailedException;

    @Positive
    public boolean isDestroyed();

    @Positive
    public String toString();

    @Positive
    public int hashCode();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);
    @Positive
}

// CFWR semantic augmentation - variant 0
