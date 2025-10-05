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
import java.util.Arrays;
import javax.crypto.SecretKey;
import javax.security.auth.Destroyable;
import javax.security.auth.DestroyFailedException;
import sun.security.util.HexDumpEncoder;
import sun.security.krb5.Asn1Exception;
import sun.security.krb5.PrincipalName;
import sun.security.krb5.EncryptionKey;
import sun.security.krb5.EncryptedData;
import sun.security.krb5.KrbException;
import sun.security.util.DerValue;

class KeyImpl implements SecretKey, Destroyable, Serializable {

    public KeyImpl(byte[] keyBytes, int keyType) {
    }

    public KeyImpl(KerberosPrincipal principal, char[] password, String algorithm) {
    }

    public final int getKeyType();

    public final String getAlgorithm();

    public final String getFormat();

    public final byte[] getEncoded();

    public void destroy() throws DestroyFailedException;

    public boolean isDestroyed();

    public String toString();

    public int hashCode();

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);
}
