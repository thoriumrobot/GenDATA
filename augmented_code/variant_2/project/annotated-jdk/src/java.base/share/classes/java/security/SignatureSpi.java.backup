/*
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.security.spec.AlgorithmParameterSpec;
import java.util.*;
import java.io.*;
import java.nio.ByteBuffer;
import sun.security.jca.JCAUtil;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class SignatureSpi {

    public SignatureSpi() {
    }

    protected SecureRandom appRandom;

    protected abstract void engineInitVerify(PublicKey publicKey) throws InvalidKeyException;

    void engineInitVerify(PublicKey publicKey, AlgorithmParameterSpec params) throws InvalidKeyException, InvalidAlgorithmParameterException;

    protected abstract void engineInitSign(PrivateKey privateKey) throws InvalidKeyException;

    protected void engineInitSign(PrivateKey privateKey, SecureRandom random) throws InvalidKeyException;

    void engineInitSign(PrivateKey privateKey, AlgorithmParameterSpec params, SecureRandom random) throws InvalidKeyException, InvalidAlgorithmParameterException;

    protected abstract void engineUpdate(byte b) throws SignatureException;

    protected abstract void engineUpdate(byte[] b, int off, int len) throws SignatureException;

    protected void engineUpdate(ByteBuffer input);

    protected abstract byte[] engineSign() throws SignatureException;

    protected int engineSign(byte[] outbuf, int offset, int len) throws SignatureException;

    protected abstract boolean engineVerify(byte[] sigBytes) throws SignatureException;

    protected boolean engineVerify(byte[] sigBytes, int offset, int length) throws SignatureException;

    @Deprecated
    protected abstract void engineSetParameter(String param, Object value) throws InvalidParameterException;

    protected void engineSetParameter(AlgorithmParameterSpec params) throws InvalidAlgorithmParameterException;

    protected AlgorithmParameters engineGetParameters();

    @Deprecated
    protected abstract Object engineGetParameter(String param) throws InvalidParameterException;

    public Object clone() throws CloneNotSupportedException;
}
