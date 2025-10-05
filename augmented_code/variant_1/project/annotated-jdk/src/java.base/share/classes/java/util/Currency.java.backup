/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.InputStream;
import java.io.IOException;
import java.io.Serializable;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.spi.CurrencyNameProvider;
import java.util.stream.Collectors;
import jdk.internal.util.StaticProperty;
import sun.util.locale.provider.CalendarDataUtility;
import sun.util.locale.provider.LocaleServiceProviderPool;
import sun.util.logging.PlatformLogger;

@AnnotatedFor({ "interning", "lock", "nullness" })
@SuppressWarnings("removal")
@UsesObjectEquals
public final class Currency implements Serializable {

    public static Currency getInstance(String currencyCode);

    public static Currency getInstance(Locale locale);

    public static Set<Currency> getAvailableCurrencies();

    public String getCurrencyCode(@GuardSatisfied Currency this);

    public String getSymbol(@GuardSatisfied Currency this);

    public String getSymbol(@GuardSatisfied Currency this, Locale locale);

    public int getDefaultFractionDigits(@GuardSatisfied Currency this);

    public int getNumericCode();

    public String getNumericCodeAsString();

    public String getDisplayName();

    public String getDisplayName(Locale locale);

    @SideEffectFree
    @Override
    public String toString(@GuardSatisfied Currency this);

    private static class CurrencyNameGetter implements LocaleServiceProviderPool.LocalizedObjectGetter<CurrencyNameProvider, String> {

        @Override
        public String getObject(CurrencyNameProvider currencyNameProvider, Locale locale, String key, Object... params);
    }

    private static class SpecialCaseEntry {
    }

    private static class OtherCurrencyEntry {
    }

    private static class CurrencyProperty {
    }
}
