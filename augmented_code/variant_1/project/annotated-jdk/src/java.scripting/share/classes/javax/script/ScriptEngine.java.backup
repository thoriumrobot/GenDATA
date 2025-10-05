/*
 * Copyright (c) 2005, 2013, Oracle and/or its affiliates. All rights reserved.
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
package javax.script;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.Reader;
import java.util.Map;
import java.util.Set;

@AnnotatedFor({ "interning" })
public interface ScriptEngine {

    @Interned
    public static final String ARGV;

    @Interned
    public static final String FILENAME;

    @Interned
    public static final String ENGINE;

    @Interned
    public static final String ENGINE_VERSION;

    @Interned
    public static final String NAME;

    @Interned
    public static final String LANGUAGE;

    @Interned
    public static final String LANGUAGE_VERSION;

    public Object eval(String script, ScriptContext context) throws ScriptException;

    public Object eval(Reader reader, ScriptContext context) throws ScriptException;

    public Object eval(String script) throws ScriptException;

    public Object eval(Reader reader) throws ScriptException;

    public Object eval(String script, Bindings n) throws ScriptException;

    public Object eval(Reader reader, Bindings n) throws ScriptException;

    public void put(String key, Object value);

    public Object get(String key);

    public Bindings getBindings(int scope);

    public void setBindings(Bindings bindings, int scope);

    public Bindings createBindings();

    public ScriptContext getContext();

    public void setContext(ScriptContext context);

    public ScriptEngineFactory getFactory();
}
