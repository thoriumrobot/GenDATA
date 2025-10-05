/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.security;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.security.spec.AlgorithmParameterSpec;
    @Positive
import java.util.*;
    @Positive
import java.io.*;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import sun.security.jca.JCAUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class SignatureSpi {

    @Positive
    public SignatureSpi() {
    @Positive
    }

    @Positive
    protected SecureRandom appRandom;

    @Positive
    protected abstract void engineInitVerify(PublicKey publicKey) throws InvalidKeyException;

    @Positive
    void engineInitVerify(PublicKey publicKey, AlgorithmParameterSpec params) throws InvalidKeyException, InvalidAlgorithmParameterException;

    @Positive
    protected abstract void engineInitSign(PrivateKey privateKey) throws InvalidKeyException;

    @Positive
    protected void engineInitSign(PrivateKey privateKey, SecureRandom random) throws InvalidKeyException;

    @Positive
    void engineInitSign(PrivateKey privateKey, AlgorithmParameterSpec params, SecureRandom random) throws InvalidKeyException, InvalidAlgorithmParameterException;

    @Positive
    protected abstract void engineUpdate(byte b) throws SignatureException;

    @Positive
    protected abstract void engineUpdate(byte[] b, int off, int len) throws SignatureException;

    @Positive
    protected void engineUpdate(ByteBuffer input);

    @Positive
    protected abstract byte[] engineSign() throws SignatureException;

    @Positive
    protected int engineSign(byte[] outbuf, int offset, int len) throws SignatureException;

    @Positive
    protected abstract boolean engineVerify(byte[] sigBytes) throws SignatureException;

    @Positive
    protected boolean engineVerify(byte[] sigBytes, int offset, int length) throws SignatureException;

    @Positive
    @Deprecated
    @Positive
    protected abstract void engineSetParameter(String param, Object value) throws InvalidParameterException;

    @Positive
    protected void engineSetParameter(AlgorithmParameterSpec params) throws InvalidAlgorithmParameterException;

    @Positive
    protected AlgorithmParameters engineGetParameters();

    @Positive
    @Deprecated
    @Positive
    protected abstract Object engineGetParameter(String param) throws InvalidParameterException;

    @Positive
    public Object clone() throws CloneNotSupportedException;
    @Positive
}
