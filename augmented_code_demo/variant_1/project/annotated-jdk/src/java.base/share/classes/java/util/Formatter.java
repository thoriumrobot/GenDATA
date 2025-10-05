/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.formatter.qual.FormatMethod;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.BufferedWriter;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.IOException;
    @Positive
import java.io.File;
    @Positive
import java.io.FileOutputStream;
    @Positive
import java.io.FileNotFoundException;
    @Positive
import java.io.Flushable;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.PrintStream;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.math.MathContext;
    @Positive
import java.math.RoundingMode;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.IllegalCharsetNameException;
    @Positive
import java.nio.charset.UnsupportedCharsetException;
    @Positive
import java.text.DateFormatSymbols;
    @Positive
import java.text.DecimalFormat;
    @Positive
import java.text.DecimalFormatSymbols;
    @Positive
import java.text.NumberFormat;
    @Positive
import java.text.spi.NumberFormatProvider;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.time.DateTimeException;
    @Positive
import java.time.Instant;
    @Positive
import java.time.ZoneId;
    @Positive
import java.time.ZoneOffset;
    @Positive
import java.time.temporal.ChronoField;
    @Positive
import java.time.temporal.TemporalAccessor;
    @Positive
import java.time.temporal.TemporalQueries;
    @Positive
import java.time.temporal.UnsupportedTemporalTypeException;
    @Positive
import jdk.internal.math.DoubleConsts;
    @Positive
import jdk.internal.math.FormattedFloatingDecimal;
    @Positive
import sun.util.locale.provider.LocaleProviderAdapter;
    @Positive
import sun.util.locale.provider.ResourceBundleBasedAdapter;

    @Positive
@AnnotatedFor({ "formatter", "index", "lock", "mustcall", "nullness" })
    @Positive
public final class Formatter implements Closeable, Flushable {

    @Positive
    public Formatter() {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias Appendable a) {
    @Positive
    }

    @Positive
    public Formatter(Locale l) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias Appendable a, Locale l) {
    @Positive
    }

    @Positive
    public Formatter(String fileName) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public Formatter(String fileName, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public Formatter(String fileName, String csn, Locale l) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public Formatter(String fileName, Charset charset, Locale l) throws IOException {
    @Positive
    }

    @Positive
    public Formatter(File file) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public Formatter(File file, String csn) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public Formatter(File file, String csn, Locale l) throws FileNotFoundException, UnsupportedEncodingException {
    @Positive
    }

    @Positive
    public Formatter(File file, Charset charset, Locale l) throws IOException {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias PrintStream ps) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias OutputStream os) {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias OutputStream os, String csn) throws UnsupportedEncodingException {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias OutputStream os, String csn, Locale l) throws UnsupportedEncodingException {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public Formatter(@MustCallAlias OutputStream os, Charset charset, Locale l) {
    @Positive
    }

    @Positive
    public Locale locale();

    @Positive
    @MustCallAlias
    @Positive
    public Appendable out(@MustCallAlias Formatter this);

    @Positive
    public String toString();

    @Positive
    public void flush();

    @Positive
    public void close();

    @Positive
    public IOException ioException();

    @Positive
    @FormatMethod
    @Positive
    @MustCallAlias
    @Positive
    public Formatter format(@MustCallAlias Formatter this, String format, Object... args);

    @Positive
    @FormatMethod
    @Positive
    @MustCallAlias
    @Positive
    public Formatter format(@MustCallAlias Formatter this, Locale l, String format, Object... args);

    @Positive
    private interface FormatString {

    @Positive
        int index();

    @Positive
        void print(Object arg, Locale l) throws IOException;

    @Positive
        String toString();
    @Positive
    }

    @Positive
    private class FixedString implements FormatString {

    @Positive
        public int index();

    @Positive
        public void print(Object arg, Locale l) throws IOException;

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public enum BigDecimalLayoutForm {

    @Positive
        SCIENTIFIC, DECIMAL_FLOAT
    @Positive
    }

    @Positive
    private class FormatSpecifier implements FormatString {

    @Positive
        public int index();

    @Positive
        public void print(Object arg, Locale l) throws IOException;

    @Positive
        public String toString();

    @Positive
        private class BigDecimalLayout {

    @Positive
            public BigDecimalLayout(BigInteger intVal, int scale, BigDecimalLayoutForm form) {
    @Positive
            }

    @Positive
            public boolean hasDot();

    @Positive
            public int scale();

    @Positive
            public StringBuilder mantissa();

    @Positive
            public StringBuilder exponent();
    @Positive
        }
    @Positive
    }

    @Positive
    private static class Flags {

    @Positive
        public int valueOf();

    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(Flags f);

    @Positive
        public Flags dup();

    @Positive
        public Flags remove(Flags f);

    @Positive
        public static Flags parse(String s, int start, int end);

    @Positive
        public static String toString(Flags f);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static class Conversion {

    @Positive
        static boolean isValid(char c);

    @Positive
        static boolean isGeneral(char c);

    @Positive
        static boolean isCharacter(char c);

    @Positive
        static boolean isInteger(char c);

    @Positive
        static boolean isFloat(char c);

    @Positive
        static boolean isText(char c);
    @Positive
    }

    @Positive
    private static class DateTime {

    @Positive
        static boolean isValid(char c);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
