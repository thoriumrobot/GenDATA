/*
    @Positive
 * Copyright (c) 2004, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.org.apache.xerces.internal.jaxp.datatype;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.org.apache.xerces.internal.util.DatatypeMessageFormatter;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.Serializable;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.math.BigInteger;
    @Positive
import java.math.RoundingMode;
    @Positive
import java.util.Calendar;
    @Positive
import java.util.Date;
    @Positive
import java.util.GregorianCalendar;
    @Positive
import java.util.Locale;
    @Positive
import java.util.TimeZone;
    @Positive
import javax.xml.datatype.DatatypeConstants;
    @Positive
import javax.xml.datatype.Duration;
    @Positive
import javax.xml.datatype.XMLGregorianCalendar;
    @Positive
import javax.xml.namespace.QName;
    @Positive
import jdk.xml.internal.SecuritySupport;

    @Positive
public class XMLGregorianCalendarImpl extends XMLGregorianCalendar implements Serializable, Cloneable {

    @Positive
    public static final XMLGregorianCalendar LEAP_YEAR_DEFAULT;

    @Positive
    protected XMLGregorianCalendarImpl(String lexicalRepresentation) throws IllegalArgumentException {
    @Positive
    }

    @Positive
    public XMLGregorianCalendarImpl() {
    @Positive
    }

    @Positive
    protected XMLGregorianCalendarImpl(BigInteger year, int month, int day, int hour, int minute, int second, BigDecimal fractionalSecond, int timezone) {
    @Positive
    }

    @Positive
    public XMLGregorianCalendarImpl(GregorianCalendar cal) {
    @Positive
    }

    @Positive
    public static XMLGregorianCalendar createDateTime(BigInteger year, int month, int day, int hours, int minutes, int seconds, BigDecimal fractionalSecond, int timezone);

    @Positive
    public static XMLGregorianCalendar createDateTime(int year, int month, int day, int hour, int minute, int second);

    @Positive
    public static XMLGregorianCalendar createDateTime(int year, int month, int day, int hours, int minutes, int seconds, int milliseconds, int timezone);

    @Positive
    public static XMLGregorianCalendar createDate(int year, int month, int day, int timezone);

    @Positive
    public static XMLGregorianCalendar createTime(int hours, int minutes, int seconds, int timezone);

    @Positive
    public static XMLGregorianCalendar createTime(int hours, int minutes, int seconds, BigDecimal fractionalSecond, int timezone);

    @Positive
    public static XMLGregorianCalendar createTime(int hours, int minutes, int seconds, int milliseconds, int timezone);

    @Positive
    public BigInteger getEon();

    @Positive
    public int getYear();

    @Positive
    public BigInteger getEonAndYear();

    @Positive
    public int getMonth();

    @Positive
    public int getDay();

    @Positive
    public int getTimezone();

    @Positive
    public int getHour();

    @Positive
    public int getMinute();

    @Positive
    public int getSecond();

    @Positive
    public int getMillisecond();

    @Positive
    public BigDecimal getFractionalSecond();

    @Positive
    public final void setYear(BigInteger year);

    @Positive
    public final void setYear(int year);

    @Positive
    public final void setMonth(int month);

    @Positive
    public final void setDay(int day);

    @Positive
    public final void setTimezone(int offset);

    @Positive
    public final void setTime(int hour, int minute, int second);

    @Positive
    public void setHour(int hour);

    @Positive
    public void setMinute(int minute);

    @Positive
    public void setSecond(int second);

    @Positive
    public final void setTime(int hour, int minute, int second, BigDecimal fractional);

    @Positive
    public final void setTime(int hour, int minute, int second, int millisecond);

    @Positive
    public int compare(XMLGregorianCalendar rhs);

    @Positive
    public XMLGregorianCalendar normalize();

    @Positive
    public static XMLGregorianCalendar parse(String lexicalRepresentation);

    @Positive
    public String toXMLFormat();

    @Positive
    public QName getXMLSchemaType();

    @Positive
    public final boolean isValid();

    @Positive
    public void add(Duration duration);

    @Positive
    private static class DaysInMonth {
    @Positive
    }

    @Positive
    public java.util.GregorianCalendar toGregorianCalendar();

    @Positive
    public GregorianCalendar toGregorianCalendar(TimeZone timezone, Locale aLocale, XMLGregorianCalendar defaults);

    @Positive
    public TimeZone getTimeZone(int defaultZoneoffset);

    @Positive
    public Object clone();

    @Positive
    public void clear();

    @Positive
    public void setMillisecond(int millisecond);

    @Positive
    public final void setFractionalSecond(BigDecimal fractional);

    @Positive
    private final class Parser {

    @Positive
        public void parse() throws IllegalArgumentException;
    @Positive
    }

    @Positive
    static BigInteger sanitize(Number value, int signum);

    @Positive
    public void reset();
    @Positive
}

// CFWR semantic augmentation - variant 1
