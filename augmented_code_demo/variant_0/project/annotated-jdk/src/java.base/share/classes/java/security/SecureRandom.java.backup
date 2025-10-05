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
package java.security;

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.math.BigInteger;
import java.util.*;
import java.util.random.RandomGenerator;
import java.util.regex.*;
import java.security.Provider.Service;
import jdk.internal.util.random.RandomSupport.RandomGeneratorProperties;
import sun.security.jca.*;
import sun.security.jca.GetInstance.Instance;
import sun.security.provider.SunEntries;
import sun.security.util.Debug;

@AnnotatedFor({ "signedness" })
@RandomGeneratorProperties(name = "SecureRandom", isStochastic = true)
public class SecureRandom extends java.util.Random {

    public SecureRandom() {
    }

    public SecureRandom(byte[] seed) {
    }

    protected SecureRandom(SecureRandomSpi secureRandomSpi, Provider provider) {
    }

    public static SecureRandom getInstance(String algorithm) throws NoSuchAlgorithmException;

    public static SecureRandom getInstance(String algorithm, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static SecureRandom getInstance(String algorithm, Provider provider) throws NoSuchAlgorithmException;

    public static SecureRandom getInstance(String algorithm, SecureRandomParameters params) throws NoSuchAlgorithmException;

    public static SecureRandom getInstance(String algorithm, SecureRandomParameters params, String provider) throws NoSuchAlgorithmException, NoSuchProviderException;

    public static SecureRandom getInstance(String algorithm, SecureRandomParameters params, Provider provider) throws NoSuchAlgorithmException;

    public final Provider getProvider();

    public String getAlgorithm();

    @Override
    public String toString();

    public SecureRandomParameters getParameters();

    public void setSeed(byte[] seed);

    @Override
    public void setSeed(long seed);

    @Override
    public void nextBytes(@PolySigned byte[] bytes);

    public void nextBytes(byte[] bytes, SecureRandomParameters params);

    @Override
    protected final int next(int numBits);

    public static byte[] getSeed(int numBytes);

    public byte[] generateSeed(int numBytes);

    private static final class StrongPatternHolder {
    }

    public static SecureRandom getInstanceStrong() throws NoSuchAlgorithmException;

    public void reseed();

    public void reseed(SecureRandomParameters params);
}
