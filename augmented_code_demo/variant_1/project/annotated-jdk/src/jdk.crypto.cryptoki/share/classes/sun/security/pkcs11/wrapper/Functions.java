/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package sun.security.pkcs11.wrapper;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import java.math.BigInteger;
    @Positive
import java.util.*;
    @Positive
import static sun.security.pkcs11.wrapper.PKCS11Constants.*;

    @Positive
public class Functions {

    @Positive
    public static String toFullHexString(long value);

    @Positive
    public static String toFullHexString(int value);

    @Positive
    public static String toHexString(@UnknownSignedness long value);

    @Positive
    public static String toHexString(@PolySigned byte[] value);

    @Positive
    public static String toBinaryString(long value);

    @Positive
    public static String toBinaryString(byte[] value);

    @Positive
    private static class Flags {

    @Positive
        String toString(long val);
    @Positive
    }

    @Positive
    public static String slotInfoFlagsToString(long flags);

    @Positive
    public static String tokenInfoFlagsToString(long flags);

    @Positive
    public static String sessionInfoFlagsToString(long flags);

    @Positive
    public static String sessionStateToString(long state);

    @Positive
    public static String mechanismInfoFlagsToString(long flags);

    @Positive
    public static long getId(Map<String, Integer> idMap, String name);

    @Positive
    public static String getMechanismName(long id);

    @Positive
    public static long getMechanismId(String name);

    @Positive
    public static String getKeyName(long id);

    @Positive
    public static long getKeyId(String name);

    @Positive
    public static String getAttributeName(long id);

    @Positive
    public static long getAttributeId(String name);

    @Positive
    public static String getObjectClassName(long id);

    @Positive
    public static long getObjectClassId(String name);

    @Positive
    public static long getHashMechId(String name);

    @Positive
    public static String getMGFName(long id);

    @Positive
    public static long getMGFId(String name);

    @Positive
    public static boolean equals(CK_DATE date1, CK_DATE date2);

    @Positive
    public static int hashCode(byte[] array);

    @Positive
    public static int hashCode(char[] array);

    @Positive
    public static int hashCode(CK_DATE date);
    @Positive
}

// CFWR semantic augmentation - variant 1
