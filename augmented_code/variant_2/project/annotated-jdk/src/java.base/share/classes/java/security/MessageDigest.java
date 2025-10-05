/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.*;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.PrintStream;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import sun.security.jca.GetInstance;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.MessageDigestSpi2;
    @Positive
import javax.crypto.SecretKey;

    @Positive
@AnnotatedFor({ "nullness", "signedness" })
    @Positive
public abstract class MessageDigest extends MessageDigestSpi {

    @Positive
    protected MessageDigest(String algorithm) {
    @Positive
    }

    @Positive
    public static MessageDigest getInstance(String algorithm) throws NoSuchAlgorithmException;

    @Positive
    public static MessageDigest getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    @Positive
    public static MessageDigest getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    @Positive
    public final Provider getProvider();

    @Positive
    public void update(byte input);

    @Positive
    public void update(@PolySigned byte[] input, int offset, int len);

    @Positive
    public void update(@PolySigned byte[] input);

    @Positive
    public final void update(ByteBuffer input);

    @Positive
    @PolySigned
    @Positive
    public byte[] digest();

    @Positive
    public int digest(@PolySigned byte[] buf, int offset, int len) throws DigestException;

    @Positive
    @PolySigned
    @Positive
    public byte[] digest(@PolySigned byte[] input);

    @Positive
    public String toString();

    @Positive
    public static boolean isEqual(byte[] digesta, byte[] digestb);

    @Positive
    public void reset();

    @Positive
    public final String getAlgorithm();

    @Positive
    public final int getDigestLength();

    @Positive
    public Object clone() throws CloneNotSupportedException;

    @Positive
    private static class Delegate extends MessageDigest implements MessageDigestSpi2 {

    @Positive
        private static final class CloneableDelegate extends Delegate implements Cloneable {
    @Positive
        }

    @Positive
        static Delegate of(MessageDigestSpi digestSpi, String algo, Provider p);

    @Positive
        @Override
    @Positive
        public Object clone() throws CloneNotSupportedException;

    @Positive
        @Override
    @Positive
        protected int engineGetDigestLength();

    @Positive
        @Override
    @Positive
        protected void engineUpdate(byte input);

    @Positive
        @Override
    @Positive
        protected void engineUpdate(@PolySigned byte[] input, int offset, int len);

    @Positive
        @Override
    @Positive
        protected void engineUpdate(ByteBuffer input);

    @Positive
        @Override
    @Positive
        public void engineUpdate(SecretKey key) throws InvalidKeyException;

    @Positive
        @Override
    @Positive
        @PolySigned
    @Positive
        protected byte[] engineDigest();

    @Positive
        @Override
    @Positive
        protected int engineDigest(@PolySigned byte[] buf, int offset, int len) throws DigestException;

    @Positive
        @Override
    @Positive
        protected void engineReset();
    @Positive
    }
    @Positive
}
