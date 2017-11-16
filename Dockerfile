FROM python:3.6
MAINTAINER Soloman Weng "soloman.weng@intellihr.com.au"
ENV REFRESHED_AT 2017-11-16

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ADD ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en

ADD . /usr/src/app
