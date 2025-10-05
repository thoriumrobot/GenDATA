/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
 */
package sun.security.pkcs11.wrapper;

import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import java.math.BigInteger;
import java.util.*;
import static sun.security.pkcs11.wrapper.PKCS11Constants.*;

public class Functions {

    public static String toFullHexString(long value);

    public static String toFullHexString(int value);

    public static String toHexString(@UnknownSignedness long value);

    public static String toHexString(@PolySigned byte[] value);

    public static String toBinaryString(long value);

    public static String toBinaryString(byte[] value);

    private static class Flags {

        String toString(long val);
    }

    public static String slotInfoFlagsToString(long flags);

    public static String tokenInfoFlagsToString(long flags);

    public static String sessionInfoFlagsToString(long flags);

    public static String sessionStateToString(long state);

    public static String mechanismInfoFlagsToString(long flags);

    public static long getId(Map<String, Integer> idMap, String name);

    public static String getMechanismName(long id);

    public static long getMechanismId(String name);

    public static String getKeyName(long id);

    public static long getKeyId(String name);

    public static String getAttributeName(long id);

    public static long getAttributeId(String name);

    public static String getObjectClassName(long id);

    public static long getObjectClassId(String name);

    public static long getHashMechId(String name);

    public static String getMGFName(long id);

    public static long getMGFId(String name);

    public static boolean equals(CK_DATE date1, CK_DATE date2);

    public static int hashCode(byte[] array);

    public static int hashCode(char[] array);

    public static int hashCode(CK_DATE date);
}
