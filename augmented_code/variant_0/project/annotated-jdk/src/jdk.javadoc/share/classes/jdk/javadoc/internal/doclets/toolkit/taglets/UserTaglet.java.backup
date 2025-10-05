/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.taglets;

import org.checkerframework.dataflow.qual.Pure;
import java.util.List;
import java.util.Set;
import javax.lang.model.element.Element;
import com.sun.source.doctree.DocTree;
import jdk.javadoc.internal.doclets.formats.html.markup.RawHtml;
import jdk.javadoc.internal.doclets.toolkit.Content;
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
import static jdk.javadoc.doclet.Taglet.Location.*;

public final class UserTaglet implements Taglet {

    public UserTaglet(jdk.javadoc.doclet.Taglet t) {
    }

    @Override
    public Set<jdk.javadoc.doclet.Taglet.Location> getAllowedLocations();

    @Override
    public boolean inField();

    @Override
    public boolean inConstructor();

    @Override
    public boolean inMethod();

    @Override
    public boolean inOverview();

    @Override
    public boolean inModule();

    @Override
    public boolean inPackage();

    @Override
    public boolean inType();

    @Pure
    @Override
    public boolean isInlineTag();

    @Override
    public boolean isBlockTag();

    @Override
    public String getName();

    @Override
    public Content getInlineTagOutput(Element element, DocTree tag, TagletWriter writer);

    @Override
    public Content getAllBlockTagOutput(Element holder, TagletWriter writer);
}
