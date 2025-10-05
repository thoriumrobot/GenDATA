/*
    @Positive
 * Copyright (c) 1996, 2019, Oracle and/or its affiliates. All rights reserved.
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
import java.util.Arrays;
    @Positive
import java.util.Properties;
    @Positive
import java.security.Key;
    @Positive
import java.security.PublicKey;
    @Positive
import java.security.KeyFactory;
    @Positive
import java.security.Security;
    @Positive
import java.security.Provider;
    @Positive
import java.security.InvalidKeyException;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.spec.InvalidKeySpecException;
    @Positive
import java.security.spec.X509EncodedKeySpec;
    @Positive
import sun.security.util.HexDumpEncoder;
    @Positive
import sun.security.util.*;

    @Positive
public class X509Key implements PublicKey {

    @Positive
    protected AlgorithmId algid;

    @Positive
    @Deprecated
    @Positive
    protected byte[] key;

    @Positive
    protected byte[] encodedKey;

    @Positive
    public X509Key() {
    @Positive
    }

    @Positive
    protected void setKey(BitArray key);

    @Positive
    protected BitArray getKey();

    @Positive
    public static PublicKey parse(DerValue in) throws IOException;

    @Positive
    protected void parseKeyBits() throws IOException, InvalidKeyException;

    @Positive
    static PublicKey buildX509Key(AlgorithmId algid, BitArray key) throws IOException, InvalidKeyException;

    @Positive
    public String getAlgorithm();

    @Positive
    public AlgorithmId getAlgorithmId();

    @Positive
    public final void encode(DerOutputStream out) throws IOException;

    @Positive
    public byte[] getEncoded();

    @Positive
    public byte[] getEncodedInternal() throws InvalidKeyException;

    @Positive
    public String getFormat();

    @Positive
    public byte[] encode() throws InvalidKeyException;

    @Positive
    public String toString();

    @Positive
    public void decode(InputStream in) throws InvalidKeyException;

    @Positive
    public void decode(byte[] encodedKey) throws InvalidKeyException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    static void encode(DerOutputStream out, AlgorithmId algid, BitArray key) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
