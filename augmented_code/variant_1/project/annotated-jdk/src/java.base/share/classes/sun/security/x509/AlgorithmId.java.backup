/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.security.*;
import sun.security.util.*;

public class AlgorithmId implements Serializable, DerEncoder {

    protected transient byte[] encodedParams;

    @Deprecated
    public AlgorithmId() {
    }

    public AlgorithmId(ObjectIdentifier oid) {
    }

    public AlgorithmId(ObjectIdentifier oid, AlgorithmParameters algparams) {
    }

    public AlgorithmId(ObjectIdentifier oid, DerValue params) throws IOException {
    }

    protected void decodeParams() throws IOException;

    public final void encode(DerOutputStream out) throws IOException;

    @Override
    public void derEncode(OutputStream out) throws IOException;

    public final byte[] encode() throws IOException;

    public final ObjectIdentifier getOID();

    public String getName();

    public AlgorithmParameters getParameters();

    public byte[] getEncodedParams() throws IOException;

    public boolean equals(AlgorithmId other);

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Override
    public boolean equals(@Nullable Object other);

    public final boolean equals(ObjectIdentifier id);

    @Override
    public int hashCode();

    protected String paramsToString();

    @Override
    public String toString();

    public static AlgorithmId parse(DerValue val) throws IOException;

    @Deprecated
    public static AlgorithmId getAlgorithmId(String algname) throws NoSuchAlgorithmException;

    public static AlgorithmId get(String algname) throws NoSuchAlgorithmException;

    public static AlgorithmId get(AlgorithmParameters algparams) throws NoSuchAlgorithmException;

    public static final ObjectIdentifier MD2_oid;

    public static final ObjectIdentifier MD5_oid;

    public static final ObjectIdentifier SHA_oid;

    public static final ObjectIdentifier SHA224_oid;

    public static final ObjectIdentifier SHA256_oid;

    public static final ObjectIdentifier SHA384_oid;

    public static final ObjectIdentifier SHA512_oid;

    public static final ObjectIdentifier SHA512_224_oid;

    public static final ObjectIdentifier SHA512_256_oid;

    public static final ObjectIdentifier SHA3_224_oid;

    public static final ObjectIdentifier SHA3_256_oid;

    public static final ObjectIdentifier SHA3_384_oid;

    public static final ObjectIdentifier SHA3_512_oid;

    public static final ObjectIdentifier DSA_oid;

    public static final ObjectIdentifier EC_oid;

    public static final ObjectIdentifier RSAEncryption_oid;

    public static final ObjectIdentifier RSASSA_PSS_oid;

    public static final ObjectIdentifier MGF1_oid;

    public static final ObjectIdentifier ed25519_oid;

    public static final ObjectIdentifier ed448_oid;

    public static final ObjectIdentifier x25519_oid;

    public static final ObjectIdentifier x448_oid;

    public static final ObjectIdentifier SHA224withECDSA_oid;

    public static final ObjectIdentifier SHA256withECDSA_oid;

    public static final ObjectIdentifier SHA384withECDSA_oid;

    public static final ObjectIdentifier SHA512withECDSA_oid;
}
