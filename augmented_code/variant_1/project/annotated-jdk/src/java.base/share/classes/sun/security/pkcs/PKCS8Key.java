/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package sun.security.pkcs;

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
import java.security.Key;
    @Positive
import java.security.KeyRep;
    @Positive
import java.security.PrivateKey;
    @Positive
import java.security.KeyFactory;
    @Positive
import java.security.MessageDigest;
    @Positive
import java.security.InvalidKeyException;
    @Positive
import java.security.NoSuchAlgorithmException;
    @Positive
import java.security.spec.InvalidKeySpecException;
    @Positive
import java.security.spec.PKCS8EncodedKeySpec;
    @Positive
import java.util.Arrays;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.security.x509.*;
    @Positive
import sun.security.util.*;

    @Positive
public class PKCS8Key implements PrivateKey {

    @Positive
    protected AlgorithmId algid;

    @Positive
    protected byte[] key;

    @Positive
    protected byte[] encodedKey;

    @Positive
    protected PKCS8Key() {
    @Positive
    }

    @Positive
    protected PKCS8Key(byte[] input) throws InvalidKeyException {
    @Positive
    }

    @Positive
    public static PrivateKey parseKey(byte[] encoded) throws IOException;

    @Positive
    public String getAlgorithm();

    @Positive
    public AlgorithmId getAlgorithmId();

    @Positive
    public byte[] getEncoded();

    @Positive
    public String getFormat();

    @Positive
    @java.io.Serial
    @Positive
    protected Object writeReplace() throws java.io.ObjectStreamException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object object);

    @Positive
    public int hashCode();

    @Positive
    public void clear();
    @Positive
}
