/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.security.x509;

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
import java.util.*;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.security.*;
    @Positive
import sun.security.util.*;

    @Positive
public class AlgorithmId implements Serializable, DerEncoder {

    @Positive
    protected transient byte[] encodedParams;

    @Positive
    @Deprecated
    @Positive
    public AlgorithmId() {
    @Positive
    }

    @Positive
    public AlgorithmId(ObjectIdentifier oid) {
    @Positive
    }

    @Positive
    public AlgorithmId(ObjectIdentifier oid, AlgorithmParameters algparams) {
    @Positive
    }

    @Positive
    public AlgorithmId(ObjectIdentifier oid, DerValue params) throws IOException {
    @Positive
    }

    @Positive
    protected void decodeParams() throws IOException;

    @Positive
    public final void encode(DerOutputStream out) throws IOException;

    @Positive
    @Override
    @Positive
    public void derEncode(OutputStream out) throws IOException;

    @Positive
    public final byte[] encode() throws IOException;

    @Positive
    public final ObjectIdentifier getOID();

    @Positive
    public String getName();

    @Positive
    public AlgorithmParameters getParameters();

    @Positive
    public byte[] getEncodedParams() throws IOException;

    @Positive
    public boolean equals(AlgorithmId other);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    @Override
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public final boolean equals(ObjectIdentifier id);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    protected String paramsToString();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public static AlgorithmId parse(DerValue val) throws IOException;

    @Positive
    @Deprecated
    @Positive
    public static AlgorithmId getAlgorithmId(String algname) throws NoSuchAlgorithmException;

    @Positive
    public static AlgorithmId get(String algname) throws NoSuchAlgorithmException;

    @Positive
    public static AlgorithmId get(AlgorithmParameters algparams) throws NoSuchAlgorithmException;

    @Positive
    public static final ObjectIdentifier MD2_oid;

    @Positive
    public static final ObjectIdentifier MD5_oid;

    @Positive
    public static final ObjectIdentifier SHA_oid;

    @Positive
    public static final ObjectIdentifier SHA224_oid;

    @Positive
    public static final ObjectIdentifier SHA256_oid;

    @Positive
    public static final ObjectIdentifier SHA384_oid;

    @Positive
    public static final ObjectIdentifier SHA512_oid;

    @Positive
    public static final ObjectIdentifier SHA512_224_oid;

    @Positive
    public static final ObjectIdentifier SHA512_256_oid;

    @Positive
    public static final ObjectIdentifier SHA3_224_oid;

    @Positive
    public static final ObjectIdentifier SHA3_256_oid;

    @Positive
    public static final ObjectIdentifier SHA3_384_oid;

    @Positive
    public static final ObjectIdentifier SHA3_512_oid;

    @Positive
    public static final ObjectIdentifier DSA_oid;

    @Positive
    public static final ObjectIdentifier EC_oid;

    @Positive
    public static final ObjectIdentifier RSAEncryption_oid;

    @Positive
    public static final ObjectIdentifier RSASSA_PSS_oid;

    @Positive
    public static final ObjectIdentifier MGF1_oid;

    @Positive
    public static final ObjectIdentifier ed25519_oid;

    @Positive
    public static final ObjectIdentifier ed448_oid;

    @Positive
    public static final ObjectIdentifier x25519_oid;

    @Positive
    public static final ObjectIdentifier x448_oid;

    @Positive
    public static final ObjectIdentifier SHA224withECDSA_oid;

    @Positive
    public static final ObjectIdentifier SHA256withECDSA_oid;

    @Positive
    public static final ObjectIdentifier SHA384withECDSA_oid;

    @Positive
    public static final ObjectIdentifier SHA512withECDSA_oid;
    @Positive
}

// CFWR semantic augmentation - variant 0
