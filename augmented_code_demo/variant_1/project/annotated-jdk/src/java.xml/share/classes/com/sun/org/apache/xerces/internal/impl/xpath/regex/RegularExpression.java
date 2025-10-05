/*
    @Positive
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xpath.regex;

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
import java.text.CharacterIterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Stack;
    @Positive
import com.sun.org.apache.xerces.internal.util.IntStack;

    @Positive
public class RegularExpression implements java.io.Serializable {

    @Positive
    public boolean matches(char[] target);

    @Positive
    public boolean matches(char[] target, int start, int end);

    @Positive
    public boolean matches(char[] target, Match match);

    @Positive
    public boolean matches(char[] target, int start, int end, Match match);

    @Positive
    public boolean matches(String target);

    @Positive
    public boolean matches(String target, int start, int end);

    @Positive
    public boolean matches(String target, Match match);

    @Positive
    public boolean matches(String target, int start, int end, Match match);

    @Positive
    boolean matchAnchor(ExpressionTarget target, Op op, Context con, int offset, int opts);

    @Positive
    public boolean matches(CharacterIterator target);

    @Positive
    public boolean matches(CharacterIterator target, Match match);

    @Positive
    static abstract class ExpressionTarget {

    @Positive
        abstract char charAt(int index);

    @Positive
        abstract boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

    @Positive
        abstract boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    @Positive
    }

    @Positive
    static final class StringTarget extends ExpressionTarget {

    @Positive
        final void resetTarget(String target);

    @Positive
        final char charAt(int index);

    @Positive
        final boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

    @Positive
        final boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    @Positive
    }

    @Positive
    static final class CharArrayTarget extends ExpressionTarget {

    @Positive
        final void resetTarget(char[] target);

    @Positive
        char charAt(int index);

    @Positive
        final boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

    @Positive
        final boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    @Positive
    }

    @Positive
    static final class CharacterIteratorTarget extends ExpressionTarget {

    @Positive
        final void resetTarget(CharacterIterator target);

    @Positive
        final char charAt(int index);

    @Positive
        final boolean regionMatches(boolean ignoreCase, int offset, int limit, String part, int partlen);

    @Positive
        final boolean regionMatches(boolean ignoreCase, int offset, int limit, int offset2, int partlen);
    @Positive
    }

    @Positive
    static final class ClosureContext {

    @Positive
        @Pure
    @Positive
        boolean contains(int offset);

    @Positive
        void reset();

    @Positive
        void addOffset(int offset);
    @Positive
    }

    @Positive
    static final class Context {

    @Positive
        void reset(CharacterIterator target, int start, int limit, int nofclosures);

    @Positive
        void reset(String target, int start, int limit, int nofclosures);

    @Positive
        void reset(char[] target, int start, int limit, int nofclosures);

    @Positive
        synchronized void setInUse(boolean inUse);
    @Positive
    }

    @Positive
    void prepare();

    @Positive
    public RegularExpression(String regex) throws ParseException {
    @Positive
    }

    @Positive
    public RegularExpression(String regex, String options) throws ParseException {
    @Positive
    }

    @Positive
    public RegularExpression(String regex, String options, Locale locale) throws ParseException {
    @Positive
    }

    @Positive
    public void setPattern(String newPattern) throws ParseException;

    @Positive
    public void setPattern(String newPattern, Locale locale) throws ParseException;

    @Positive
    public void setPattern(String newPattern, String options) throws ParseException;

    @Positive
    public void setPattern(String newPattern, String options, Locale locale) throws ParseException;

    @Positive
    public String getPattern();

    @Positive
    public String toString();

    @Positive
    public String getOptions();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    boolean equals(String pattern, int options);

    @Positive
    public int hashCode();

    @Positive
    public int getNumberOfGroups();
    @Positive
}

// CFWR semantic augmentation - variant 1
