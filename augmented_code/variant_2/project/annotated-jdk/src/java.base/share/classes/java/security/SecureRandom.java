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
package java.security;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.*;
    @Positive
import java.util.random.RandomGenerator;
    @Positive
import java.util.regex.*;
    @Positive
import java.security.Provider.Service;
    @Positive
import jdk.internal.util.random.RandomSupport.RandomGeneratorProperties;
    @Positive
import sun.security.jca.*;
    @Positive
import sun.security.jca.GetInstance.Instance;
    @Positive
import sun.security.provider.SunEntries;
    @Positive
import sun.security.util.Debug;

    @Positive
@AnnotatedFor({ "signedness" })
    @Positive
@RandomGeneratorProperties(name = "SecureRandom", isStochastic = true)
    @Positive
public class SecureRandom extends java.util.Random {

    @Positive
    public SecureRandom() {
    @Positive
    }

    @Positive
    public SecureRandom(byte[] seed) {
    @Positive
    }

    @Positive
    protected SecureRandom(SecureRandomSpi secureRandomSpi, Provider provider) {
    @Positive
    }

    @Positive
    public static SecureRandom getInstance(String algorithm) throws NoSuchAlgorithmException;

    @Positive
    public static SecureRandom getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    @Positive
    public static SecureRandom getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    @Positive
    public static SecureRandom getInstance(String algorithm, SecureRandomParameters params) throws NoSuchAlgorithmException;

    @Positive
    public static SecureRandom getInstance(String algorithm, SecureRandomParameters params, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    @Positive
    public static SecureRandom getInstance(String algorithm, SecureRandomParameters params, Provider provider) throws NoSuchAlgorithmException;

    @Positive
    public final Provider getProvider();

    @Positive
    public String getAlgorithm();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public SecureRandomParameters getParameters();

    @Positive
    public void setSeed(byte[] seed);

    @Positive
    @Override
    @Positive
    public void setSeed(long seed);

    @Positive
    @Override
    @Positive
    public void nextBytes(@PolySigned byte[] bytes);

    @Positive
    public void nextBytes(byte[] bytes, SecureRandomParameters params);

    @Positive
    @Override
    @Positive
    protected final int next(int numBits);

    @Positive
    public static byte[] getSeed(int numBytes);

    @Positive
    public byte[] generateSeed(int numBytes);

    @Positive
    private static final class StrongPatternHolder {
    @Positive
    }

    @Positive
    public static SecureRandom getInstanceStrong() throws NoSuchAlgorithmException;

    @Positive
    public void reseed();

    @Positive
    public void reseed(SecureRandomParameters params);
    @Positive
}
