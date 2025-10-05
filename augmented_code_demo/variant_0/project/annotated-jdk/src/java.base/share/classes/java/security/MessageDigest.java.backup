/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.*;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import sun.security.jca.GetInstance;
import sun.security.util.Debug;
import sun.security.util.MessageDigestSpi2;
import javax.crypto.SecretKey;

@AnnotatedFor({ "nullness", "signedness" })
public abstract class MessageDigest extends MessageDigestSpi {

    protected MessageDigest(String algorithm) {
    }

    public static MessageDigest getInstance(String algorithm) throws NoSuchAlgorithmException;

    public static MessageDigest getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static MessageDigest getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    public final Provider getProvider();

    public void update(byte input);

    public void update(@PolySigned byte[] input, int offset, int len);

    public void update(@PolySigned byte[] input);

    public final void update(ByteBuffer input);

    @PolySigned
    public byte[] digest();

    public int digest(@PolySigned byte[] buf, int offset, int len) throws DigestException;

    @PolySigned
    public byte[] digest(@PolySigned byte[] input);

    public String toString();

    public static boolean isEqual(byte[] digesta, byte[] digestb);

    public void reset();

    public final String getAlgorithm();

    public final int getDigestLength();

    public Object clone() throws CloneNotSupportedException;

    private static class Delegate extends MessageDigest implements MessageDigestSpi2 {

        private static final class CloneableDelegate extends Delegate implements Cloneable {
        }

        static Delegate of(MessageDigestSpi digestSpi, String algo, Provider p);

        @Override
        public Object clone() throws CloneNotSupportedException;

        @Override
        protected int engineGetDigestLength();

        @Override
        protected void engineUpdate(byte input);

        @Override
        protected void engineUpdate(@PolySigned byte[] input, int offset, int len);

        @Override
        protected void engineUpdate(ByteBuffer input);

        @Override
        public void engineUpdate(SecretKey key) throws InvalidKeyException;

        @Override
        @PolySigned
        protected byte[] engineDigest();

        @Override
        protected int engineDigest(@PolySigned byte[] buf, int offset, int len) throws DigestException;

        @Override
        protected void engineReset();
    }
}
