/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.security.auth.kerberos;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.*;
import java.util.Date;
import java.util.Arrays;
import java.net.InetAddress;
import java.util.Objects;
import javax.crypto.SecretKey;
import javax.security.auth.Refreshable;
import javax.security.auth.Destroyable;
import javax.security.auth.RefreshFailedException;
import javax.security.auth.DestroyFailedException;
import sun.security.util.HexDumpEncoder;

public class KerberosTicket implements Destroyable, Refreshable, java.io.Serializable {

    public KerberosTicket(byte[] asn1Encoding, KerberosPrincipal client, KerberosPrincipal server, byte[] sessionKey, int keyType, boolean[] flags, Date authTime, Date startTime, Date endTime, Date renewTill, InetAddress[] clientAddresses) {
    }

    public final KerberosPrincipal getClient();

    public final KerberosPrincipal getServer();

    public final SecretKey getSessionKey();

    public final int getSessionKeyType();

    public final boolean isForwardable();

    public final boolean isForwarded();

    public final boolean isProxiable();

    public final boolean isProxy();

    public final boolean isPostdated();

    public final boolean isRenewable();

    public final boolean isInitial();

    public final boolean[] getFlags();

    public final java.util.Date getAuthTime();

    public final java.util.Date getStartTime();

    public final java.util.Date getEndTime();

    public final java.util.Date getRenewTill();

    public final java.net.InetAddress[] getClientAddresses();

    public final byte[] getEncoded();

    public boolean isCurrent();

    public void refresh() throws RefreshFailedException;

    public void destroy() throws DestroyFailedException;

    public boolean isDestroyed();

    public String toString();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);
}
